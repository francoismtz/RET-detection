"""Le chemin d'inference RAPIDE du U-Net : les memes entrees, calculees sans detour.

Objectif : ~1 ms par cellule tout compris. Le profil du chemin de reference se
lisait ainsi, par crop :

    masque de cellule                 0.44 ms
    densite optique + canaux (15)     1.2  ms
    mise en tenseur (HWC -> CHW)      0.075 ms
    reseau (unet, lot 64, fp32)       0.08 ms

Le reseau n'etait donc pas le probleme : c'etaient les canaux et le masque. Ce
module ne change RIEN au resultat (0 pixel de difference sur 77 106 contre le
chemin de reference), il change la facon de le calculer :

  * le fond (95e percentile) par `np.partition` et non par `np.percentile` ;
  * le masque de cellule sans logarithme : OD moyenne > 0.04 equivaut a
    produit des transmittances < 10^-0.12 ;
  * les canaux de transmittance (`t_*`) directement depuis l'image ;
  * la pile ecrite DIRECTEMENT en CHW dans un tampon de lot prealloue et
    epingle (pinned) : une seule copie vers le GPU ;
  * les quatre canaux relatifs a la cellule (z robustes de c_nmb, de l'OD
    totale, du cosinus NMB, et rang de c_nmb) calcules SUR LE GPU, en lot. En
    repli CPU, ils ne sont calcules que sur les pixels de la cellule ;
  * demi-precision optionnelle sur le GPU (`demi`).

    ex = rapide.charger("models/modele_rapide_t8.pt")
    probas, cellules, cadres = ex(rgbs, boxes, medianes=[med] * len(rgbs))

`medianes` est la loi de lame (med_phys[9], med_rgb[3]) de chaque crop, mesuree
sur l'acquisition elle-meme (`maturite.loi_de_lame`).
"""
import cv2
import numpy as np
import torch

from . import noyau as N
from . import reseau as R

SEUIL_PRODUIT = np.float32(10.0 ** (-3 * 0.04))     # 3 canaux x seuil OD 0.04
_K3 = np.ones((3, 3), np.uint8)
_T = ("t_r", "t_g", "t_b")
_I = ("i_r", "i_g", "i_b")
_RAPIDES = set(_T) | set(_I) | set(N.NOMS_CELL) | {"dist_bord", "cellule"}
_U_NMB = (N.STAIN_NMB / np.linalg.norm(N.STAIN_NMB)).astype(np.float32)
_PINV_NMB = N._PINV[1].astype(np.float32)          # ligne « NMB » de la pseudo-inverse
_PINV_HGB = N._PINV[0].astype(np.float32)          # ligne « hemoglobine »

# Les reglages d'architecture, par defaut. Le checkpoint porte les siens
# (`d["params"]`), qui l'emportent : ceci ne sert qu'a un fichier incomplet.
PARAMS = dict(taille=96, archi="unet2", filtres=16, norm_couches="bn",
              dropout=0.0, dilatation_fond=1, separable=False, residuel=False,
              norm_lame="mediane", appareil="auto")


# --------------------------------------------------------------------------
#  1. Fond et masque, sans logarithme
# --------------------------------------------------------------------------
def fond(rgb):
    """95e percentile par canal, identique a `noyau.estimate_background`
    (interpolation lineaire de NumPy), par selection partielle."""
    flat = rgb.reshape(-1, 3)
    n = flat.shape[0]
    q = 0.95 * (n - 1)
    lo = int(np.floor(q))
    if lo + 1 >= n:
        return flat.max(axis=0) + N.EPS
    part = np.partition(flat, (lo, lo + 1), axis=0)
    a, b = part[lo], part[lo + 1]
    return (a + (b - a) * np.float32(q - lo)) + N.EPS


def transmittance(rgb, i0):
    """I/I0, avec le meme plancher que `optical_density` (1e-3)."""
    t = np.maximum(rgb, np.float32(1e-3))
    t /= i0
    return t


def masque_cellule(rgb, box, i0=None, t=None):
    """« composante x boite », identique a `noyau.masque_cellule` : le seuil
    sur l'OD moyenne est pris sur le produit des transmittances."""
    if t is None:
        t = transmittance(rgb, fond(rgb) if i0 is None else i0)
    p = t[..., 0] * t[..., 1]
    p *= t[..., 2]
    m = (p < SEUIL_PRODUIT).view(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K3)
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m |= (1 - ff)
    n, lab = cv2.connectedComponents(m)
    x, y, bw, bh = box
    cy = min(max(int(y + bh / 2), 0), h - 1)
    cx = min(max(int(x + bw / 2), 0), w - 1)
    et = lab[cy, cx]
    comp = (lab == et) if et != 0 else m.astype(bool)
    return comp & N.disque_boite(m.shape, box)


# --------------------------------------------------------------------------
#  2. La pile, ecrite en CHW dans un tampon de lot
# --------------------------------------------------------------------------
def _bord(noms, i0, med):
    """Valeur de chaque canal HORS crop (la ou l'OD de reference vaut 0)."""
    v = np.zeros(len(noms), np.float32)
    for k, n in enumerate(noms):
        if n in _T:
            i = _T.index(n)
            v[k] = (1.0 - (med[1][i] if med is not None else 1.0)) * N.ECHELLE_OD
        elif n in _I:
            i = _I.index(n)
            v[k] = (i0[i] - 1.0) * N.ECHELLE_OD
    return v


def _stats_cellule(t, cell, noms):
    """Les 4 canaux relatifs a la cellule, calcules sur les SEULS pixels de la
    cellule : z robustes (mediane / MAD) de c_nmb, de l'OD totale et du cosinus
    NMB, et rang de c_nmb dans la cellule (0-1)."""
    sel = cell.astype(bool)
    nsel = int(sel.sum())
    if nsel < 20:
        return {n: None for n in noms}
    od = -np.log10(t[sel])                              # nsel x 3, float32
    src = {}
    if "z_nmb" in noms or "rang_nmb" in noms:
        src["c_nmb"] = od @ _PINV_NMB
    if "z_tot" in noms:
        src["od_tot"] = od.sum(axis=1)
    if "z_cos" in noms:
        src["cos_nmb"] = (od @ _U_NMB) / (np.sqrt((od * od).sum(axis=1)) + 1e-3)
    out = {}
    for n in noms:
        if n == "rang_nmb":
            r = np.empty(nsel, np.float32)
            r[np.argsort(src["c_nmb"], kind="stable")] = \
                np.arange(nsel, dtype=np.float32) / max(nsel - 1, 1)
            out[n] = r
        else:
            vs = src[{"z_nmb": "c_nmb", "z_tot": "od_tot", "z_cos": "cos_nmb"}[n]]
            med = _mediane(vs)
            mad = 1.4826 * _mediane(np.abs(vs - med)) \
                + (0.05 if n == "z_cos" else 0.01)
            out[n] = np.clip((vs - med) / mad, -10, 10) * 0.25
    return out                                          # valeurs SUR LA CELLULE


def _mediane(v):
    """`np.median` d'un vecteur 1D, par selection partielle."""
    n = v.shape[0]
    h = n // 2
    if n % 2:
        return np.partition(v, h)[h]
    p = np.partition(v, (h - 1, h))
    return np.mean(p[h - 1:h + 1])


def _ecrire_pile(dst, noms, t, i0, cell, med, y0, x0, sauf=()):
    """`dst` : vue (C, T, T) du tampon, deja remplie de la valeur de bord.
    `sauf` : canaux laisses tels quels (calcules ailleurs)."""
    h, w = cell.shape
    sl = (slice(y0, y0 + h), slice(x0, x0 + w))
    cel = [n for n in noms if n in N.NOMS_CELL and n not in sauf]
    stats = _stats_cellule(t, cell, cel) if cel else {}
    sel = cell.astype(bool) if cel else None
    for k, n in enumerate(noms):
        if n in sauf:
            continue
        if n in stats:
            vue = dst[k][sl]
            vue[...] = 0.0
            if stats[n] is not None:
                vue[sel] = stats[n]
            continue
        if n in _T:
            i = _T.index(n)
            m = med[1][i] if med is not None else 1.0
            dst[k][sl] = (t[..., i] - m) * N.ECHELLE_OD
        elif n in _I:
            i = _I.index(n)
            dst[k][sl] = (t[..., i] * np.float32(i0[i]) - 1.0) * N.ECHELLE_OD
        elif n == "cellule":
            dst[k][sl] = cell
        elif n == "dist_bord":
            dst[k][sl] = N.distance_bord(cell) / 10.0
        else:
            raise KeyError(n)


class Extracteur:
    """Le modele charge (3 reseaux moyennes), et un chemin d'inference en lot."""

    def __init__(self, chemin, dev=None, demi=False, lot=256, stats_gpu=None,
                 taille=None):
        """`demi` : reseaux en demi-precision. `stats_gpu` : les 4 canaux
        relatifs a la cellule calcules sur le GPU, en lot (par defaut : oui des
        qu'il y a un GPU). `taille` : cote du carre (par defaut celui de
        l'entrainement, 96) ; un crop plus grand demande un extracteur a la
        taille superieure, multiple de 16."""
        d = torch.load(chemin, map_location="cpu", weights_only=False)
        P = dict(PARAMS)
        P.update(d["params"])
        self.P = P
        self.dev = dev or R.appareil(P.get("appareil", "auto"))
        self.noms = tuple(d["noms_canaux"])
        if not all(n in _RAPIDES for n in self.noms):
            raise RuntimeError(f"canaux {self.noms} : pas de chemin rapide")
        self.seuil = float(d["seuil"])
        self.taille = int(taille or P["taille"])
        self.norm_lame = P["norm_lame"] == "mediane"
        self.demi = bool(demi) and self.dev.type == "cuda"
        self.lot = int(lot)
        self.modeles = []
        for w in d["poids"]:
            m = R.UNet(len(self.noms), P["archi"], int(P["filtres"]),
                       P["norm_couches"], float(P["dropout"]),
                       int(P["dilatation_fond"]), bool(P["separable"]),
                       bool(P["residuel"]))
            m.load_state_dict(w)
            m = m.to(self.dev).eval()
            if self.demi:
                m = m.half()
            self.modeles.append(m)
        T = self.taille
        self._tampon = torch.empty((self.lot, len(self.noms), T, T),
                                   dtype=torch.float32,
                                   pin_memory=(self.dev.type == "cuda"))
        self._np = self._tampon.numpy()
        # les canaux relatifs a la cellule, sur le GPU : le CPU ne fournit que
        # la transmittance (bord = 1, comme une OD nulle) et le masque
        self._cel = [n for n in self.noms if n in N.NOMS_CELL]
        self.stats_gpu = (bool(self._cel) and self.dev.type == "cuda"
                          if stats_gpu is None else bool(stats_gpu))
        if self.stats_gpu:
            self._tampon_t = torch.empty((self.lot, 3, T, T), dtype=torch.float32,
                                         pin_memory=(self.dev.type == "cuda"))
            self._tampon_c = torch.empty((self.lot, T, T), dtype=torch.float32,
                                         pin_memory=(self.dev.type == "cuda"))
            self._np_t, self._np_c = self._tampon_t.numpy(), self._tampon_c.numpy()
            self._idx_cel = [self.noms.index(n) for n in self._cel]
            self._pinv = torch.as_tensor(_PINV_NMB, device=self.dev)
            self._pinv_hgb = torch.as_tensor(_PINV_HGB, device=self.dev)
            self._u = torch.as_tensor(_U_NMB, device=self.dev)
        else:
            # repli CPU : l'hemoglobine integree est calculee crop par crop
            self._hgb_cpu = np.zeros((self.lot, 2), np.float64)
        self._extras = None
        # PAS de `cudnn.benchmark` : il relance un auto-reglage a chaque taille
        # de lot nouvelle (un lot par image, de taille variable) et coute alors
        # des dizaines de ms par lot, pour aucun gain mesure.
        torch.backends.cudnn.benchmark = False

    # ---- un crop -> sa place dans le tampon ------------------------------
    def _prepare(self, j, rgb, box, med):
        T = self.taille
        h, w = rgb.shape[:2]
        if max(h, w) > T:
            raise ValueError(f"crop {h}x{w} plus grand que {T}")
        y0, x0 = (T - h) // 2, (T - w) // 2
        i0 = fond(rgb)
        t = transmittance(rgb, i0)
        cell = masque_cellule(rgb, box, t=t)
        dst = self._np[j]
        dst[:] = _bord(self.noms, i0, med)[:, None, None]
        cf = cell.astype(np.float32)
        _ecrire_pile(dst, self.noms, t, i0, cf, med, y0, x0,
                     sauf=self._cel if self.stats_gpu else ())
        cellp = np.zeros((T, T), bool)
        cellp[y0:y0 + h, x0:x0 + w] = cell
        if self.stats_gpu:
            sl = (slice(y0, y0 + h), slice(x0, x0 + w))
            dt = self._np_t[j]
            dt[:] = 1.0
            for i in range(3):
                dt[i][sl] = t[..., i]
            dc = self._np_c[j]
            dc[:] = 0.0
            dc[sl] = cf
        else:
            # meme definition que `_stats_gpu` : c_hgb deconvoluee, sommee sur
            # les pixels de la cellule
            nsel = int(cell.sum())
            tot = float((-np.log10(t[cell]) @ _PINV_HGB).sum()) if nsel else 0.0
            self._hgb_cpu[j] = (tot, nsel)
        return cellp, (y0, x0, h, w)

    # ---- les 4 canaux relatifs a la cellule, sur le GPU -------------------
    @torch.no_grad()
    def _stats_gpu(self, b, n):
        """Memes definitions que `_stats_cellule` : z robustes (mediane / MAD)
        de c_nmb, de l'OD totale et du cosinus NMB, et rang de c_nmb dans la
        cellule. Les medianes et le rang passent par un tri STABLE en lot, les
        pixels hors cellule renvoyes a +inf : les egalites gardent l'ordre ligne
        par ligne, comme `np.argsort(kind="stable")`."""
        t = self._tampon_t[:n].to(self.dev, non_blocking=True)
        cell = self._tampon_c[:n].to(self.dev, non_blocking=True)
        B, T = cell.shape[0], cell.shape[1]
        sel = (cell > 0.5).flatten(1)                          # (B, T*T)
        nsel = sel.sum(1)                                      # (B,)
        valide = (nsel >= 20).float()[:, None]
        inf = torch.full_like(cell.flatten(1), float("inf"))
        od = -torch.log10(t)                                   # (B, 3, T, T)
        # l'hemoglobine integree sur la cellule (c_hgb deconvoluee) : l'ordonnee
        # du scattergramme, l'analogue du FSC. Gratuite ici, l'OD est deja la.
        hgb = (od * self._pinv_hgb[None, :, None, None]).sum(1).flatten(1)
        tot = torch.where(sel, hgb, torch.zeros_like(hgb)).sum(1)
        self._extras = dict(
            hgb_totale=tot.cpu().numpy(),
            hgb_moyenne=(tot / nsel.clamp(min=1).float()).cpu().numpy(),
            n_cellule=nsel.cpu().numpy())
        src = {}
        if "z_nmb" in self._cel or "rang_nmb" in self._cel:
            src["c_nmb"] = (od * self._pinv[None, :, None, None]).sum(1)
        if "z_tot" in self._cel:
            src["od_tot"] = od.sum(1)
        if "z_cos" in self._cel:
            src["cos_nmb"] = ((od * self._u[None, :, None, None]).sum(1)
                              / (torch.sqrt((od * od).sum(1)) + 1e-3))
        i1 = ((nsel - 1) // 2).clamp(min=0)[:, None]
        i2 = (nsel // 2).clamp(max=T * T - 1)[:, None]

        def mediane(v):                                        # v (B, T*T)
            s, _ = torch.sort(torch.where(sel, v, inf), dim=1)
            return (s.gather(1, i1) + s.gather(1, i2)) / 2     # (B, 1)

        for k, nom in zip(self._idx_cel, self._cel):
            if nom == "rang_nmb":
                v = src["c_nmb"].flatten(1)
                _, ordre = torch.sort(torch.where(sel, v, inf), dim=1, stable=True)
                rang = torch.empty_like(ordre)
                rang.scatter_(1, ordre, torch.arange(T * T, device=self.dev)
                              .expand(B, -1))
                o = rang.float() / (nsel - 1).clamp(min=1).float()[:, None]
            else:
                v = src[{"z_nmb": "c_nmb", "z_tot": "od_tot",
                         "z_cos": "cos_nmb"}[nom]].flatten(1)
                med = mediane(v)
                mad = (1.4826 * mediane((v - med).abs())
                       + (0.05 if nom == "z_cos" else 0.01))
                o = ((v - med) / mad).clamp(-10, 10) * 0.25
            o = torch.where(sel, o, torch.zeros_like(o)) * valide
            b[:, k] = o.view(B, T, T).to(b.dtype)

    @torch.no_grad()
    def _passe(self, n):
        b = self._tampon[:n].to(self.dev, non_blocking=True)
        if self.demi:
            b = b.half()
        if self.stats_gpu:
            self._stats_gpu(b, n)
        else:
            tot, nsel = self._hgb_cpu[:n, 0], self._hgb_cpu[:n, 1]
            self._extras = dict(
                hgb_totale=tot.astype(np.float32),
                hgb_moyenne=(tot / np.maximum(nsel, 1)).astype(np.float32),
                n_cellule=nsel.astype(np.int64))
        s = None
        for m in self.modeles:
            p = torch.sigmoid(m(b))[:, 0]
            s = p if s is None else s + p
        return (s / len(self.modeles)).float().cpu().numpy()

    def __call__(self, rgbs, boxes, medianes=None, extras=False):
        """-> (probas, masques de cellule, cadres), tous sur le carre T x T ;
        avec `extras=True`, un quatrieme element : dict de vecteurs par crop
        (`hgb_totale`, `hgb_moyenne`, `n_cellule`).

        `medianes` : le couple (med_phys[9], med_rgb[3]) de chaque crop -- la
        loi de lame, mesuree sur l'acquisition elle-meme."""
        if self.norm_lame and medianes is None:
            raise ValueError("ce modele demande la loi de lame (`medianes`)")
        probas, cellules, cadres, sup = [], [], [], []
        n = len(rgbs)
        for d in range(0, n, self.lot):
            k = min(self.lot, n - d)
            for j in range(k):
                med = medianes[d + j] if self.norm_lame else None
                cp, cadre = self._prepare(j, rgbs[d + j], boxes[d + j], med)
                cellules.append(cp)
                cadres.append(cadre)
            probas.extend(self._passe(k))
            if extras:
                sup.append(self._extras)
        if extras:
            sup = {c: np.concatenate([s[c] for s in sup]) if sup else np.zeros(0)
                   for c in ("hgb_totale", "hgb_moyenne", "n_cellule")}
            return probas, cellules, cadres, sup
        return probas, cellules, cadres

    def masques(self, rgbs, boxes, medianes=None):
        """Masques de reticulum, recadres a la taille de chaque crop."""
        pr, ce, ca = self(rgbs, boxes, medianes)
        out = []
        for p, c, (y0, x0, h, w) in zip(pr, ce, ca):
            out.append(((p >= self.seuil) & c)[y0:y0 + h, x0:x0 + w])
        return out


def charger(chemin, **kw):
    return Extracteur(chemin, **kw)
