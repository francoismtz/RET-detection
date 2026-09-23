"""
Etape 4a -- le `Preclassifier_output.xml`, au format Vision Hub, augmente du %RET et de l'IRF.
=============================================================================================

**Le format est celui que Vision Hub lit et ecrit**, releve sur son writer, sur
son parseur, et surtout sur une analyse RET reelle de son ecosysteme -- pas
devine. Cinq details le separent du `Recognition.xml` que l'automate depose a
cote de ses images, et chacun casse la lecture EN SILENCE s'il est manque :

  1. **le fichier s'appelle `Preclassifier_output.xml`** : Vision Hub reconnait
     un dossier d'analyse a ce fichier, pas au nom du dossier ;
  2. **la cellule est `<Cell>`, pas `<Point>`** : des `<Point>` donnent zero
     cellule, sans erreur ;
  3. **le score par classe est `<MultiClass_Score>`, au SINGULIER** ;
  4. **`<Output_Counts>` existe**, avec une entree par categorie presente, plus
     `All` et `All WBC` a zero ;
  5. seules les categories NON VIDES sont ecrites.

**RET et RBC sont lus comme RET et RBC**, sous les identifiants de l'enum
`CategoryType` du moteur de reconnaissance : `14 Reticulocyte`, `52 RBC`,
`51 WBC`, `12 Debris`, `100 Colision`.

**La bizarrerie des colonnes est reproduite volontairement.** Les quatre
`<MultiClass_Score>` portent les noms du modele WBC -- Lymphocyte, Monocyte,
Neutrophil, Eosinophils -- quel que soit le modele tourne. Un debris culmine
donc sur « Lymphocyte » et un reticulocyte sur « Monocyte ». C'est faux et
c'est le format : le corriger produirait un fichier que Vision Hub ne saurait
plus relire. La correspondance reelle est 1=Debris 2=Reticulocyte 3=WBC 4=RBC.

CE QUI EST AJOUTE, et signale comme tel :

  * un bloc `<Analysis>` : le pourcentage de reticulocytes, l'IRF, la porte qui
    les separe, et les comptes dont ils sortent ;
  * sur chaque `<Cell>` de globule rouge mesure, deux attributs `DOUCE` et
    `IMMATURE`.

Les deux sont invisibles pour Vision Hub, dont le parseur cherche des balises
et des attributs precis et ignore le reste (verifie en lui faisant lire le
fichier).

**Le %RET et l'IRF ne sont pas la meme chose.** Le %RET est un rapport de
comptes du detecteur : il ne se calibre pas. L'IRF est la part des
reticulocytes au-dela d'une porte posee sur `douce` (`config.PORTE_IRF`).
"""
from __future__ import annotations

import datetime as _dt
import os
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from .config import PORTE_IRF

SEUIL_SCORE = 20            # Algo_Threshold_Score, celui de l'automate
MAX_CELLS = 1000            # Algo_MaxCells, comme l'analyse RET de reference

# nom canonique -> (identifiant CategoryType, nom XML)
CATEGORIES = {
    "Debris": (12, "Debris"),
    "Reticulocyte": (14, "Reticulocyte"),
    "WBC": (51, "WBC"),
    "RBC": (52, "RBC"),
    "Collision": (100, "Colision"),
    "Unclassified": (11, "Unclassified"),
    "Stuck cell": (9, "Stuck cell"),
}
# l'ordre de `<Output_Counts>` de l'analyse RET de reference
ORDRE_SORTIE = ("Debris", "Reticulocyte", "WBC", "RBC", "Colision")
COLONNES_MULTICLASS = ("Lymphocyte", "Monocyte", "Neutrophil", "Eosinophils")

NOM_FICHIER = "Preclassifier_output.xml"


def resume(cel, grandeurs, porte=PORTE_IRF):
    """-> le dictionnaire de resultats : %RET, IRF, et les comptes dont ils sortent.

    %RET  = reticulocytes / globules rouges retenus. Un rapport de comptes.
    IRF   = part des reticulocytes dont `douce` depasse la porte.
    """
    ret = cel.est_ret
    n_rouge, n_ret = len(cel), int(ret.sum())
    douce = grandeurs["douce"]
    mesurable = np.isfinite(douce)

    d_ret = douce[ret & mesurable]
    n_ret_mes = int(d_ret.size)
    n_immature = int((d_ret > porte).sum())
    return dict(
        n_detectees=int(sum(len(d) for d in cel.toutes)),
        n_rouges=n_rouge,
        n_rbc=int((~ret).sum()),
        n_ret=n_ret,
        n_ret_mesures=n_ret_mes,
        n_immatures=n_immature,
        n_non_mesurables=int((~mesurable).sum()),
        n_couture=int(cel.couture.sum()),
        pct_ret=100.0 * n_ret / max(n_rouge, 1),
        irf=(100.0 * n_immature / n_ret_mes) if n_ret_mes else float("nan"),
        porte_irf=float(porte),
        douce_mediane_ret=float(np.median(d_ret)) if n_ret_mes else float("nan"),
        comptes_bruts=cel.comptes(),
    )


def _attr(**kw):
    return "".join(f" {k}={quoteattr(str(v))}" for k, v in kw.items())


def ecrit(chemin, cel, grandeurs, res, modele, bench="",
          modele_maturite="modele_rapide_t8.pt"):
    """Ecrit le Preclassifier_output.xml. -> le chemin ecrit."""
    # ou chaque cellule retenue se trouve dans le tableau des grandeurs
    ou = {}
    for u, (k, x, y) in enumerate(zip(cel.image, cel.cx, cel.cy)):
        ou[(int(k), int(x), int(y))] = u

    total = res["comptes_bruts"]
    comptees = sum(total.values())

    lignes = ['<?xml version="1.0" encoding="utf-8"?>', "<Root>"]
    lignes.append("<Recognition_Parameters>")
    lignes.append(f"<Param Algo_ModelName={quoteattr(modele)} />")
    lignes.append(f'<Param Algo_Threshold_Score="{SEUIL_SCORE}" />')
    lignes.append('<Param Algo_Threshold_HighScore="0" />')
    lignes.append(f'<Param Algo_MaxCells="{MAX_CELLS}" />')
    lignes.append(f"<Param Bench={quoteattr(bench or '')} />")
    lignes.append(f'<Param Counted_cells="{comptees}" />')
    lignes.append("</Recognition_Parameters>")

    lignes.append("<Output_Counts>")
    for canon in ORDRE_SORTIE:
        nom_xml = CATEGORIES.get(
            "Collision" if canon == "Colision" else canon, (0, canon))[1]
        cle = "Collision" if canon == "Colision" else canon
        lignes.append(f'<Output Category="{escape(nom_xml)}" '
                      f'Count="{total.get(cle, 0)}" />')
    lignes.append('<Output Category="All" Count="0" />')
    lignes.append('<Output Category="All WBC" Count="0" />')
    lignes.append("</Output_Counts>")
    lignes.append("<Flags />")

    lignes.append("<Pictures>")
    ident = 0
    for k, dets in enumerate(cel.toutes):
        pid = "".join(c for c in os.path.splitext(cel.fichiers[k])[0]
                      if c.isdigit()) or f"{k + 1:02d}"
        lignes.append(f'<Picture ID={quoteattr(pid)}>')
        par_classe = {}
        for d in dets:
            par_classe.setdefault(d.label, []).append(d)
        # seules les categories NON VIDES, dans l'ordre des identifiants
        for canon in sorted(par_classe, key=lambda c: CATEGORIES.get(c, (999,))[0]):
            cat_id, nom_xml = CATEGORIES.get(canon, (11, "Unclassified"))
            cellules = par_classe[canon]
            lignes.append(f'<Category ID="{cat_id}" Name="{escape(nom_xml)}" '
                          f'Count="{len(cellules)}">')
            for d in cellules:
                ident += 1
                sup = ""
                u = ou.get((k, int(d.center_x), int(d.center_y)))
                if u is not None and np.isfinite(grandeurs["douce"][u]):
                    douce = float(grandeurs["douce"][u])
                    sup = _attr(DOUCE=f"{douce:.4f}",
                                IMMATURE=int(douce > res["porte_irf"]))
                lignes.append(
                    f'<Cell ID="{ident}" X="{int(d.center_x)}" '
                    f'Y="{int(d.center_y)}" WIDTH="{int(d.width)}" '
                    f'HEIGHT="{int(d.height)}" SCORE="{d.score:.8g}"{sup}>')
                mc = list(d.multiclass) + [0.0] * 4
                for nom_col, v in zip(COLONNES_MULTICLASS, mc):
                    lignes.append(f'<MultiClass_Score Category="{nom_col}" '
                                  f'Score="{float(v):.8g}" />')
                lignes.append("</Cell>")
            lignes.append("</Category>")
        lignes.append("</Picture>")
    lignes.append("</Pictures>")

    # ---- le bloc ajoute --------------------------------------------------
    lignes.append("<Analysis>")
    lignes.append("<!-- Bloc AJOUTE par ret_inference. Vision Hub l'ignore : son "
                  "parseur cherche des balises precises. -->")
    lignes.append(
        f'<Result Name="RET_Percent" Value="{res["pct_ret"]:.4f}" Unit="%" '
        f'Numerator="{res["n_ret"]}" Denominator="{res["n_rouges"]}" '
        f'Definition="reticulocytes / globules rouges retenus, classes par le '
        f'detecteur" />')
    lignes.append(
        f'<Result Name="IRF" Value="{res["irf"]:.4f}" Unit="%" '
        f'Numerator="{res["n_immatures"]}" Denominator="{res["n_ret_mesures"]}" '
        f'Gate="douce &gt; {res["porte_irf"]:.4f}" '
        f'Definition="part des reticulocytes dont la somme de probabilite de '
        f'reticulum sur la cellule depasse la porte" />')
    lignes.append(
        f'<Selection Detected="{res["n_detectees"]}" '
        f'RedCells="{res["n_rouges"]}" RBC="{res["n_rbc"]}" '
        f'Reticulocytes="{res["n_ret"]}" Measured="{res["n_ret_mesures"]}" '
        f'Unmeasurable="{res["n_non_mesurables"]}" '
        f'OnSeam="{res["n_couture"]}" '
        f'Rule="RBC et Reticulocyte, boite de 20 a 90 px, a plus de 3 px du '
        f'bord" />')
    lignes.append(
        f'<Maturity Model={quoteattr(modele_maturite)} Quantity="douce" '
        f'MedianReticulocyte="{res["douce_mediane_ret"]:.4f}" '
        f'Note="DOUCE et IMMATURE sont portes par chaque Cell mesuree" />')
    lignes.append(f'<Generated By="ret_inference" '
                  f'At="{_dt.datetime.now().isoformat(timespec="seconds")}" />')
    lignes.append("</Analysis>")
    lignes.append("</Root>")

    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("".join(lignes))
    return chemin
