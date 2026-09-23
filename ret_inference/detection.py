"""
Etape 1 -- EDD4d_RET-V3.3.000 sur le GPU, et la selection des globules rouges.
=============================================================================

Le post-traitement du detecteur (decoupe verticale en deux moities de 2048,
ordre `[ymin, xmin, ymax, xmax]`, indices de classe 1-bases, seuil a 20 %,
resolution des doublons) vit dans `edd4.py`, ou il est documente.

Ce module ajoute :

  * **le GPU**. La CUDA EP fait passer une image de 3,4 s a 0,25 s (0,15 s avec
    `coupe=True`). Les replis silencieux d'onnxruntime sur le CPU sont fermes
    dans `edd4.load_session` ;
  * **la selection**, decidee a UN SEUL endroit : trois passes (extraction,
    appariement a la verite terrain, statistiques) relisent les memes
    detections et s'alignent par INDICE. Deux filtres ecrits deux fois qui
    divergent d'une cellule decaleraient tout ce qui suit sans erreur.

CE QUI EST GARDE pour la mesure de maturite : les classes `RBC` et
`Reticulocyte`, hors boites de moins de 20 px ou de plus de 90 px de cote, hors
cellules collees au bord (tronquees : leur masque et leur fond seraient faux).
C'est le denominateur clinique du pourcentage de reticulocytes. Les debris et
les leucocytes sont detectes, ecrits dans le XML, et ne comptent pas.

CE QUI EST GARDE MAIS SIGNALE : la couture. Une cellule a cheval sur la jointure
des deux demi-images n'est vue entiere par aucune des deux. L'effet est petit
(~1 %) et touche numerateur et denominateur du meme cote ; il est marque
(`couture`) plutot que retire, pour qu'on puisse le verifier.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import config
from .edd4 import DEFAULT_MIN_SCORE, load_session, predict_full_image

# ---- la selection, a une seule place ----------------------------------------
COTE_MIN, COTE_MAX = 20, 90   # le filtre du jeu d'evaluation, garde a l'identique
GARDE_BORD = 3                # px : une cellule plus pres du bord est tronquee
LARGEUR, HAUTEUR = 4096, 2160
LARGEUR_DEMI = 2048           # la couture du modele `EDD4d`
BANDE_COUTURE = 64
MARGE = 10                    # contexte autour de la boite, vu par le U-Net

PROVIDERS_GPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]

_SESSIONS = {}


def prepare_dll_cuda():
    """Rend visibles les DLL CUDA que `torch` embarque.

    torch cu128 porte dans `torch/lib` cudart 12, cuBLAS, cuFFT, cuRAND, cuDNN 9
    et nvrtc -- exactement ce qu'onnxruntime-gpu 1.25 va chercher. En les lui
    montrant, la chaine tourne dans UN seul environnement avec UNE seule pile
    CUDA : pas de roues `nvidia-*` en double, pas deux cuDNN qui se disputent le
    meme nom de DLL.
    """
    try:
        import torch
        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except (ImportError, OSError, AttributeError):
        pass
    try:
        import onnxruntime as ort
        ort.preload_dlls()
    except Exception:                                          # noqa: BLE001
        pass


def lister_images(dossier, motif=config.MOTIF_IMAGES):
    """Les images d'une acquisition, triees (Picture01, Picture02, ...)."""
    import fnmatch
    return sorted(f for f in os.listdir(dossier)
                  if fnmatch.fnmatch(f.lower(), motif.lower()))


class Cellules:
    """Les detections d'une acquisition, filtrees et dans un ordre fige.

    L'ordre est `(image, y, x)` et jamais le score : deux passes qui trient
    autrement ne s'alignent plus par indice.
    """

    __slots__ = ("nom", "fichiers", "toutes", "image", "classe", "score",
                 "x0", "y0", "w", "h", "cx", "cy", "couture", "multiclass",
                 "garde")

    def __init__(self, nom, fichiers, par_image):
        self.nom, self.fichiers = nom, fichiers
        self.toutes = par_image                    # [[Detection, ...], ...]

        img, cl, sc, cx, cy, w, h, mc = [], [], [], [], [], [], [], []
        for k, dets in enumerate(par_image):
            for d in dets:
                img.append(k)
                cl.append(d.label)
                sc.append(d.score)
                cx.append(d.center_x)
                cy.append(d.center_y)
                w.append(d.width)
                h.append(d.height)
                mc.append(d.multiclass)
        img = np.array(img, np.int32)
        cl = np.array(cl, "U16")
        w, h = np.array(w, np.int32), np.array(h, np.int32)
        cx, cy = np.array(cx, np.int32), np.array(cy, np.int32)
        # le coin, avec l'arithmetique de l'instrument (un plancher, pas un
        # arrondi) -- la convention avec laquelle les crops d'apprentissage ont
        # ete exportes
        x0 = cx - w // 2 - (w % 2)
        y0 = cy - h // 2 - (h % 2)

        rouge = (cl == "RBC") | (cl == "Reticulocyte")
        taille = ((w >= COTE_MIN) & (w <= COTE_MAX)
                  & (h >= COTE_MIN) & (h <= COTE_MAX))
        dedans = ((x0 > GARDE_BORD) & (y0 > GARDE_BORD)
                  & (x0 + w < LARGEUR - GARDE_BORD)
                  & (y0 + h < HAUTEUR - GARDE_BORD))
        garde = np.flatnonzero(rouge & taille & dedans)
        garde = garde[np.lexsort((x0[garde], y0[garde], img[garde]))]

        self.garde = garde
        self.image, self.classe, self.score = img[garde], cl[garde], np.array(sc)[garde]
        self.x0, self.y0, self.w, self.h = x0[garde], y0[garde], w[garde], h[garde]
        self.cx, self.cy = cx[garde], cy[garde]
        self.multiclass = [mc[i] for i in garde]
        self.couture = (np.abs(self.x0 - LARGEUR_DEMI) < BANDE_COUTURE) | (
            np.abs(self.x0 + self.w - LARGEUR_DEMI) < BANDE_COUTURE)

    def __len__(self):
        return len(self.image)

    @property
    def est_ret(self):
        return self.classe == "Reticulocyte"

    def indices_image(self, k):
        return np.flatnonzero(self.image == k)

    def crop_marge(self, rgb, j):
        """Le crop avec sa marge de contexte, et la boite DANS ce crop.

        `MARGE` px autour de la boite : c'est ce que le U-Net a vu a
        l'entrainement, et le masque de cellule a besoin d'un peu de fond pour
        estimer le sien.
        """
        a = max(0, int(self.x0[j]) - MARGE)
        b = max(0, int(self.y0[j]) - MARGE)
        c = min(rgb.shape[1], int(self.x0[j] + self.w[j]) + MARGE)
        e = min(rgb.shape[0], int(self.y0[j] + self.h[j]) + MARGE)
        return rgb[b:e, a:c], (int(self.x0[j]) - a, int(self.y0[j]) - b,
                               int(self.w[j]), int(self.h[j]))

    def comptes(self):
        """Les comptes bruts, toutes classes, avant filtre -- pour le XML."""
        c = {}
        for dets in self.toutes:
            for d in dets:
                c[d.label] = c.get(d.label, 0) + 1
        return c


def charge_session(modele=None, gpu=True, coupe=False, threads=None):
    """-> la session du detecteur, en cache (utile en traitement par lot)."""
    modele = os.path.abspath(modele or config.modele_detection())
    config.verifie_modele(modele, "detection")
    cle = (modele, bool(gpu), bool(coupe and gpu), threads)
    if cle not in _SESSIONS:
        if gpu:
            prepare_dll_cuda()
        if coupe and gpu:
            from .coupe_gpu import charge_coupee
            session = charge_coupee(modele, providers_tronc=PROVIDERS_GPU,
                                    threads=threads)
        else:
            session = load_session(modele, threads=threads,
                                   providers=PROVIDERS_GPU if gpu else None)
        _SESSIONS[cle] = session
    return _SESSIONS[cle]


def detecte(dossier_images, nom, modele=None, gpu=True, threads=None,
            verbose=True, coupe=False, motif=config.MOTIF_IMAGES):
    """-> (Cellules, chronos). Une passe EDD4 sur toutes les images du dossier.

    `coupe=True` passe par `coupe_gpu` : le tronc sur le GPU, les
    NonMaxSuppression et leur post-traitement sur le CPU. Environ 1,5 fois plus
    rapide, pour des cellules identiques. Le prix est deux fichiers ONNX derives
    a garder en coherence, d'ou le defaut a False.
    """
    fichiers = lister_images(dossier_images, motif)
    if not fichiers:
        raise SystemExit(f"aucune image '{motif}' dans {dossier_images}")

    t0 = time.time()
    session = charge_session(modele, gpu=gpu, coupe=coupe, threads=threads)
    t_charge = time.time() - t0
    if verbose:
        print(f"  detecteur  : {os.path.basename(modele or config.modele_detection())}")
        print(f"  providers  : {session.get_providers()}")
        print(f"  session    : {t_charge:.2f} s   |   {len(fichiers)} images, "
              f"seuil {DEFAULT_MIN_SCORE:.0f} %")

    t0 = time.time()
    par_image, durees = [], []
    for k, f in enumerate(fichiers):
        t1 = time.time()
        par_image.append(predict_full_image(session,
                                            os.path.join(dossier_images, f)))
        durees.append(time.time() - t1)
        if verbose and (k + 1) % 20 == 0:
            print(f"    {k + 1}/{len(fichiers)} images, "
                  f"{sum(len(d) for d in par_image)} detections, "
                  f"{time.time() - t0:.1f} s")
    t_det = time.time() - t0

    cel = Cellules(nom, fichiers, par_image)
    chrono = dict(t_charge=t_charge, t_detection=t_det,
                  t_par_image=float(np.median(durees[1:] or durees)),
                  t_premier=float(durees[0]),
                  providers=list(session.get_providers()))
    if verbose:
        brut = sum(len(d) for d in par_image)
        print(f"  detection  : {t_det:.1f} s pour {len(fichiers)} images "
              f"({chrono['t_par_image']:.2f} s/image, mediane)")
        print(f"  cellules   : {brut} detectees, {len(cel)} globules rouges "
              f"retenus ({int(cel.est_ret.sum())} reticulocytes)")
    return cel, chrono
