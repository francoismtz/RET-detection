"""
Etape 4b -- le scattergramme RET, nu, a la facon du XN.
======================================================

Le XN trace chaque globule rouge dans un plan :

    abscisse  SFL -- fluorescence laterale, donc la teneur en ARN ;
    ordonnee  FSC -- diffusion vers l'avant, qui suit le volume et la charge en
                     hemoglobine.

**L'abscisse est la grandeur qui porte la porte, et pas une autre.** C'est
`douce`, la somme de la probabilite de reticulum sur la cellule -- exactement ce
que la porte compare. Tracer une grandeur et en seuiller une autre montrerait un
recouvrement qui n'est pas celui du seuil. La frontiere LFR / MFR+HFR est donc
verticale, sans chevauchement **par construction**, comme sur le XN.
L'ordonnee est l'hemoglobine integree sur la cellule, l'analogue du FSC.

Reproduire le SFL du XN est hors de portee -- c'est une fluorescence mesuree
dans un flux calibre, nous lisons une densite optique sur un frottis -- et ce
n'est pas necessaire : ce qui doit etre comparable, c'est la FORME du nuage et
la position de la porte dedans, pas l'unite des axes. D'ou **l'image nue** :
ni titre, ni graduation, ni unite, ni legende, pour se poser a cote des
scattergrammes du XN sans rien qui detonne.

CE QUI SEPARE PAR CONSTRUCTION ET CE QUI NE SEPARE PAS. La porte coupe
l'abscisse : magenta a gauche, bordeaux a droite. En revanche la frontiere
hematie / reticulocyte est celle d'EDD4, qui ne regarde pas cette grandeur : le
nuage bleu recouvre donc les magenta. Ce recouvrement ne dit rien de la porte,
mais il dit quelque chose du modele d'extraction : il GRADUE la maturite, il ne
CLASSE pas RBC / RET (utilise comme classeur, F1 0,73 contre 0,82 pour EDD4).

Couleurs du XN : bleu = hematie mure, magenta = reticulocyte sous la porte
(LFR), bordeaux = reticulocyte au-dela (MFR+HFR).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

# les teintes relevees dans les scattergrammes du XN
BLEU, MAGENTA, BORDEAUX = "#1414C8", "#B4189B", "#8C1414"
CADRE = "#8B1A8B"        # le violet des axes du XN
# Pas d'echantillonnage en pratique : 25 000 points ne coutent rien a tracer, et
# le XN montre TOUTES ses cellules -- c'est la densite du nuage qui fait sa lecture.
N_POINTS = 200000
RNG = np.random.default_rng(3)


def _echantillonne(n):
    if n <= N_POINTS:
        return np.arange(n)
    return RNG.choice(n, N_POINTS, replace=False)


def scattergramme(cel, grandeurs, res, sortie, dpi=150, cote=6.4):
    """L'image nue : le nuage, le cadre en equerre, rien d'autre."""
    douce = grandeurs["douce"]
    hgb = grandeurs["hgb_totale"]
    ok = np.isfinite(douce) & np.isfinite(hgb)
    ret = cel.est_ret & ok
    porte = res["porte_irf"]

    fig = plt.figure(figsize=(cote, cote), facecolor="white")
    # l'equerre du XN touche les bords : on lui laisse juste de quoi respirer
    ax = fig.add_axes([0.085, 0.075, 0.895, 0.905])
    ax.set_facecolor("white")

    # les hematies d'abord, les reticulocytes par-dessus : ils sont 25 fois
    # moins nombreux et c'est eux qu'on vient lire
    for masque, couleur, taille, alpha in (
            ((~cel.est_ret) & ok, BLEU, 3.2, 0.80),
            (ret & (douce <= porte), MAGENTA, 7.0, 1.0),
            (ret & (douce > porte), BORDEAUX, 7.0, 1.0)):
        idx = np.flatnonzero(masque)
        if not len(idx):
            continue
        pris = idx[_echantillonne(len(idx))]
        ax.scatter(douce[pris], hgb[pris], s=taille, c=couleur, alpha=alpha,
                   linewidths=0, marker=".")

    # ---- le cadre du XN : deux cotes, des crans, aucun chiffre -----------
    v = douce[ok]
    haut_x = float(np.percentile(v, 99.9)) if len(v) else 1.0
    w = hgb[ok]
    haut_y = float(np.percentile(w, 99.9)) if len(w) else 1.0
    ax.set_xlim(-0.02 * haut_x, 1.06 * haut_x)
    ax.set_ylim(-0.02 * haut_y, 1.10 * haut_y)
    for cote_ax in ("top", "right"):
        ax.spines[cote_ax].set_visible(False)
    for cote_ax in ("left", "bottom"):
        ax.spines[cote_ax].set_color(CADRE)
        ax.spines[cote_ax].set_linewidth(2.6)
    ax.set_xticks(np.linspace(0, haut_x, 6))
    ax.set_yticks(np.linspace(0, haut_y, 6))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(color=CADRE, width=2.6, length=9, direction="out")
    ax.grid(False)

    os.makedirs(os.path.dirname(os.path.abspath(sortie)), exist_ok=True)
    fig.savefig(sortie, dpi=dpi, facecolor="white")
    plt.close(fig)
    return sortie
