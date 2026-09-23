"""Physique, masque de cellule et loi de lame -- la partie inference seulement.

Repere physique : une lame coloree est un milieu absorbant. Beer-Lambert donne
OD = -log10(I/I0), lineaire en concentration, et deux colorants s'ADDITIONNENT
en OD. Une ombre multiplie I par s < 1, donc ajoute log10(1/s) a chaque canal :
c'est un deplacement le long de l'axe gris, achromatique, alors qu'un colorant a
un vecteur spectral propre. La deconvolution a deux colorants (Ruifrok &
Johnston, 2001) separe ainsi l'hemoglobine du nouveau bleu de methylene (NMB).

Ce fichier est extrait du module d'entrainement du U-Net : les fonctions sont
identiques, seules l'augmentation et la construction des piles d'apprentissage
ont ete retirees.
"""
import cv2
import numpy as np

EPS = 1e-6
MARGE_BOITE = 1.15
ECHELLE_OD = 4.0            # les canaux en OD valent quelques dixiemes -> O(1)

# vecteurs de colorant (OD par canal R, G, B)
STAIN_HGB = np.array([0.470, 0.452, 0.758], np.float32)
STAIN_NMB = np.array([0.548, 0.837, 0.000], np.float32)
_E1 = np.array([1, -1, 0], np.float32) / np.sqrt(2)
_E2 = np.array([1, 1, -2], np.float32) / np.sqrt(6)

# les 9 grandeurs physiques de la loi de lame, puis les 4 canaux relatifs a la
# cellule que le reseau recoit
NOMS_PHYS = ("od_r", "od_g", "od_b", "c_hgb", "c_nmb", "od_tot",
             "chroma1", "chroma2", "cos_nmb", "dist_bord", "cellule")
NOMS_CELL = ("z_nmb", "z_tot", "z_cos", "rang_nmb")


# --------------------------------------------------------------------------
#  1. Densite optique, deconvolution
# --------------------------------------------------------------------------
def estimate_background(rgb, pct=95.0):
    return np.percentile(rgb.reshape(-1, 3), pct, axis=0) + EPS


def optical_density(rgb, i0):
    return -np.log10(np.clip(rgb, 1e-3, None) / i0)


def od(rgb, i0=None):
    if i0 is None:
        i0 = estimate_background(rgb)
    return optical_density(rgb, i0).astype(np.float32)


def _pinv():
    v = np.stack([STAIN_HGB, STAIN_NMB]).astype(np.float32)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + EPS)
    return np.linalg.pinv(v.T)


_PINV = _pinv()             # ligne 0 : hemoglobine, ligne 1 : NMB


def deconv(od3):
    h, w = od3.shape[:2]
    c = od3.reshape(-1, 3) @ _PINV.T
    return c[:, 0].reshape(h, w), c[:, 1].reshape(h, w)


def chroma(od3):
    return od3 - od3.mean(axis=2, keepdims=True)


def cosinus_nmb(od3, eps=1e-3):
    n = np.linalg.norm(od3, axis=2) + eps
    return (od3 @ (STAIN_NMB / np.linalg.norm(STAIN_NMB))) / n


def distance_bord(cellule):
    return cv2.distanceTransform(cellule.astype(np.uint8), cv2.DIST_L2, 3)


# --------------------------------------------------------------------------
#  2. Masque de cellule (« composante x boite »)
# --------------------------------------------------------------------------
def fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return (m | (1 - ff)).astype(bool)


def segment_cell(rgb, box, seuil_od=0.04, i0=None):
    ctx_od = optical_density(rgb, estimate_background(rgb) if i0 is None
                             else i0).mean(axis=2)
    m = (ctx_od > seuil_od).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = fill_holes(m.astype(bool)).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    h, w = m.shape
    x, y, bw, bh = box
    cy = min(max(int(y + bh / 2), 0), h - 1)
    cx = min(max(int(x + bw / 2), 0), w - 1)
    et = lab[cy, cx]
    return (lab == et) if et != 0 else m.astype(bool)


def disque_boite(shape, box, marge=MARGE_BOITE):
    """L'ellipse inscrite dans la boite du detecteur, elargie de `marge`."""
    h, w = shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2.0, y + bh / 2.0
    a, b = marge * bw / 2.0, marge * bh / 2.0
    dx = ((np.arange(w) - cx) / a) ** 2
    dy = ((np.arange(h) - cy) / b) ** 2
    return (dy[:, None] + dx[None, :]) <= 1.0


def masque_cellule(rgb, box, i0=None):
    """« composante x boite » : la composante connexe sous le centre de la boite,
    limitee a l'ellipse de la boite."""
    return segment_cell(rgb, box, i0=i0) & disque_boite(rgb.shape, box)


# --------------------------------------------------------------------------
#  3. Loi de lame : la mediane des hematies de l'acquisition
# --------------------------------------------------------------------------
class _Atelier:
    """Calcule les cartes physiques d'un crop une seule fois, a la demande."""

    __slots__ = ("od3", "cell", "_c", "_ch")

    def __init__(self, od3, cell):
        self.od3, self.cell = od3, cell
        self._c = self._ch = None

    def brut(self, nom):
        return np.asarray(self._brut(nom), np.float32)

    def _brut(self, nom):
        if nom == "od_r":
            return self.od3[..., 0]
        if nom == "od_g":
            return self.od3[..., 1]
        if nom == "od_b":
            return self.od3[..., 2]
        if nom in ("c_hgb", "c_nmb"):
            if self._c is None:
                self._c = deconv(self.od3)
            return self._c[0 if nom == "c_hgb" else 1]
        if nom == "od_tot":
            return self.od3.sum(axis=2)
        if nom in ("chroma1", "chroma2"):
            if self._ch is None:
                self._ch = chroma(self.od3)
            return self._ch @ (_E1 if nom == "chroma1" else _E2)
        if nom == "cos_nmb":
            return cosinus_nmb(self.od3)
        raise KeyError(nom)


def medianes_depuis(crops):
    """`crops` : iterable de (rgb, masque_cellule) -> (med_phys[9], med_rgb[3]).

    `med_rgb` est la transmittance mediane par canal : c'est elle que le reseau
    soustrait a ses trois canaux image. `med_phys` est gardee pour la forme du
    contrat (le modele a 15 canaux la lisait)."""
    ordre = NOMS_PHYS[:9]
    vp, vr = [], []
    for rgb, cell in crops:
        o = od(rgb)
        a = _Atelier(o, cell)
        vp.append(np.stack([a.brut(n)[cell] for n in ordre], axis=1))
        vr.append(np.power(10.0, -o).astype(np.float32)[cell])
    if not vp:
        return np.zeros(9, np.float32), np.ones(3, np.float32)
    return (np.median(np.concatenate(vp), axis=0).astype(np.float32),
            np.median(np.concatenate(vr), axis=0).astype(np.float32))
