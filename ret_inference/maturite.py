"""
Etapes 2 et 3 -- la loi de lame, puis le U-Net sur tous les globules rouges.
===========================================================================

Le reseau est le modele RAPIDE : `unet3`, **8 canaux** -- l'image normalisee
par son fond (I/I0, trois canaux), les quatre canaux relatifs a la cellule (z
robustes de la concentration NMB, de l'OD totale et du cosinus NMB, rang de la
concentration NMB) et le masque de cellule -- trois reseaux moyennes, sans TTA,
seuil 0,5. Il est charge par `rapide.charger`, qui reconstruit l'architecture
depuis le fichier lui-meme, liste des canaux comprise. 0,8 ms par cellule sur
GPU, preparation des entrees comprise.

**La loi de lame.** Le reseau lit l'image normalisee par son fond, moins la
mediane par canal des HEMATIES de l'acquisition elle-meme -- 400, tirees une
image sur cinq pour couvrir la derive de coloration le long du frottis, jamais
plus de 40 par image. C'est la seule calibration de la chaine, et elle ne
regarde aucune annotation.

**Les grandeurs.** `douce` est la somme de la probabilite de reticulum sur les
pixels de la cellule -- c'est elle que la porte de l'IRF compare. Une somme de
probabilites reste proportionnelle sur une cellule faiblement reticulee, la ou
un seuil rendrait zero. `hgb_totale` est l'hemoglobine integree sur la cellule
(concentration deconvoluee, Beer-Lambert a deux colorants) : l'analogue du FSC
du XN, et l'ordonnee du scattergramme. Les amplitudes en OD (`quantite`,
`intensite`, `pic`, `totale`, `totale_propre`) ne sont plus calculees : elles ne
servaient ni a l'IRF ni au %RET. Les champs restent, a NaN, pour que le `.npz`
garde sa forme.

Le lot est traite par cotes homogenes (96 ou 112, multiple de 16) : un
extracteur par cote.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import config

PAS_CALIB = 5           # une image sur cinq pour la loi de lame
N_PAR_IMAGE = 40        # hematies au plus par image de calibration
N_LAME = 400            # hematies retenues au total
GRAINE = 0
LOT = 256               # cellules envoyees d'un coup sur le GPU

CHAMPS = ("aire_cellule", "aire", "fraction", "quantite", "intensite",
          "pic", "totale", "totale_propre", "douce", "hgb_totale",
          "hgb_moyenne")
CHAMPS_ABANDONNES = ("quantite", "intensite", "pic", "totale", "totale_propre")

_CACHE = {}


def _cote(h, w):
    """Le cote du carre qui recoit un crop h x w (multiple de 16, 96 au moins)."""
    return max(96, int(np.ceil(max(h, w) / 16.0)) * 16)


def charge_reseau(modele=None, verbose=True, appareil="auto", cote=96):
    """-> l'extracteur rapide pour un cote de carre. En cache par cote.

    `appareil="cpu"` sert au temoin `--sur-cpu` : sans lui, seul le detecteur
    serait debranche et le U-Net continuerait sur la carte.
    """
    modele = modele or config.modele_maturite()
    config.verifie_modele(modele, "maturite")
    cle = (os.path.abspath(modele), appareil, int(cote))
    if cle in _CACHE:
        return _CACHE[cle]
    from . import rapide as Q
    from . import reseau as R
    dev = R.appareil(appareil)
    ex = Q.charger(modele, dev=dev, lot=LOT, taille=int(cote))
    _CACHE[cle] = ex
    if verbose:
        print(f"  extracteur : {os.path.basename(modele)}  |  "
              f"{len(ex.modeles)} reseaux, {len(ex.noms)} canaux "
              f"({', '.join(ex.noms)}), seuil {ex.seuil:.2f}, sans TTA")
        print(f"  appareil   : {dev}  |  canaux cellule sur "
              f"{'GPU' if ex.stats_gpu else 'CPU'}  |  carre {cote}")
    return ex


def _lit(chemin):
    import cv2
    img = cv2.imdecode(np.fromfile(chemin, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"image illisible : {chemin}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def loi_de_lame(cel, dossier, verbose=True):
    """La mediane par canal des hematies de CETTE acquisition.

    Tirees une image sur `PAS_CALIB` : la coloration derive le long du frottis,
    et une calibration lue sur les seules premieres images serait celle du debut
    de la lame, pas de la lame.
    """
    from . import noyau as N
    t0 = time.time()
    rng = np.random.default_rng(GRAINE)
    crops = []
    for k in range(0, len(cel.fichiers), PAS_CALIB):
        js = cel.indices_image(k)
        js = js[~cel.est_ret[js]]                      # des hematies, pas des RET
        if len(js) == 0:
            continue
        if len(js) > N_PAR_IMAGE:
            js = rng.choice(js, N_PAR_IMAGE, replace=False)
        rgb = _lit(os.path.join(dossier, cel.fichiers[k]))
        for j in js:
            crops.append(cel.crop_marge(rgb, int(j)))
    if len(crops) < 60:
        raise SystemExit(f"{len(crops)} hematies seulement : trop peu pour une "
                         f"loi de lame (il en faut 60 au minimum)")
    rng.shuffle(crops)
    lot = crops[:N_LAME]

    def _pair(r, b):
        r = np.asarray(r, np.float32)
        if r.max() > 1.5:
            r = r / 255.0
        return r, N.masque_cellule(r, b)

    med = N.medianes_depuis(_pair(r, b) for r, b in lot)
    dt = time.time() - t0
    if verbose:
        print(f"  loi de lame: {len(lot)} hematies sur "
              f"{len(range(0, len(cel.fichiers), PAS_CALIB))} images   {dt:.1f} s")
    return med, dict(n_lame=len(lot), n_candidats=len(crops), t_loi_lame=dt)


def _mesure_groupe(items, med, modele, appareil):
    """items = [(rgb, box)] -> [(masque, grandeurs)], par le chemin rapide."""
    rgbs = []
    for rgb, _ in items:
        rgb = np.asarray(rgb, np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgbs.append(rgb)
    boxes = [b for _, b in items]

    # ---- extraction, par cote de carre ------------------------------------
    n = len(items)
    par_cote = {}
    for i, r in enumerate(rgbs):
        par_cote.setdefault(_cote(*r.shape[:2]), []).append(i)
    probas, cellules, cadres = [None] * n, [None] * n, [None] * n
    hgb_t = np.full(n, np.nan, np.float32)
    hgb_m = np.full(n, np.nan, np.float32)
    seuil = 0.5
    for cote, idx in par_cote.items():
        ex = charge_reseau(modele, verbose=False, appareil=appareil, cote=cote)
        seuil = ex.seuil
        pr, ce, ca, sup = ex([rgbs[i] for i in idx], [boxes[i] for i in idx],
                             medianes=[med] * len(idx), extras=True)
        for u, i in enumerate(idx):
            probas[i], cellules[i], cadres[i] = pr[u], ce[u], ca[u]
            hgb_t[i], hgb_m[i] = sup["hgb_totale"][u], sup["hgb_moyenne"][u]

    # ---- grandeurs --------------------------------------------------------
    sorties = []
    for i, (pr, cellp, (y0, x0, h, w)) in enumerate(zip(probas, cellules, cadres)):
        aire_c = int(cellp.sum())
        if aire_c < 40:
            # un masque de moins de 40 px n'est pas une cellule : la mesurer
            # produirait des grandeurs sur du bruit
            sorties.append((None, {c: np.nan for c in CHAMPS}))
            continue
        m = (pr >= seuil) & cellp
        aire = int(m.sum())
        d = dict(aire_cellule=aire_c, aire=aire, fraction=aire / float(aire_c),
                 douce=float(pr[cellp].sum()),
                 hgb_totale=float(hgb_t[i]), hgb_moyenne=float(hgb_m[i]))
        d.update({c: np.nan for c in CHAMPS_ABANDONNES})
        sorties.append((m[y0:y0 + h, x0:x0 + w], d))
    return sorties


def mesure(cel, dossier, med, modele=None, verbose=True, appareil="auto"):
    """-> (grandeurs, chronos). Une entree par globule rouge retenu."""
    modele = modele or config.modele_maturite()
    ex = charge_reseau(modele, verbose=False, appareil=appareil)
    n = len(cel)
    out = {c: np.full(n, np.nan, np.float32) for c in CHAMPS}
    t0 = time.time()
    for k in range(len(cel.fichiers)):
        js = [int(j) for j in cel.indices_image(k)]
        if not js:
            continue
        rgb = _lit(os.path.join(dossier, cel.fichiers[k]))
        for d in range(0, len(js), LOT):
            bloc = js[d:d + LOT]
            items = [cel.crop_marge(rgb, j) for j in bloc]
            for j, (_m, mes) in zip(bloc, _mesure_groupe(items, med, modele,
                                                         appareil)):
                for c in CHAMPS:
                    out[c][j] = mes[c]
        if verbose and (k + 1) % 20 == 0:
            fait = int(np.isfinite(out["douce"]).sum())
            print(f"    {k + 1}/{len(cel.fichiers)} images, {fait} cellules "
                  f"mesurees, {time.time() - t0:.1f} s")
    dt = time.time() - t0
    mesurees = int(np.isfinite(out["douce"]).sum())
    if verbose:
        print(f"  maturite   : {dt:.1f} s pour {n} cellules "
              f"({1000 * dt / max(n, 1):.2f} ms/cellule)   "
              f"{n - mesurees} non mesurables")
    return out, dict(t_maturite=dt, n_cellules=n, n_mesurees=mesurees,
                     ms_par_cellule=1000 * dt / max(n, 1),
                     seuil_unet=float(ex.seuil), tta=False,
                     stats_gpu=bool(ex.stats_gpu), appareil=str(ex.dev),
                     modele_maturite=os.path.basename(modele))
