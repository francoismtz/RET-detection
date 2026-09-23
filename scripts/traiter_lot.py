"""La chaine sur toutes les acquisitions d'un dossier -- reprise sur interruption.

Cherche, sous RACINE, tous les dossiers qui contiennent des `Picture*.jpg`
(le dossier brut de l'automate, ou son sous-dossier `pictures/`) et les traite
un par un avec `chaine.traite`. Les modeles sont charges une seule fois. Une
acquisition deja traitee (resume.json present) est sautee : le lot se reprend.

    python scripts/traiter_lot.py RACINE --sortie sorties/ [--coupe] [--sans-figure]

Sorties : <sortie>/<acquisition>/{resume.json, cellules.npz, scattergramme_RET.png,
          <dossier d'analyse>/Preclassifier_output.xml}
          <sortie>/journal_lot.txt, <sortie>/resultats_lot.csv
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ret_inference import config                       # noqa: E402
from ret_inference.chaine import nom_acquisition, traite   # noqa: E402


def acquisitions(racine, motif):
    """{nom: dossier} de tous les dossiers qui contiennent des images."""
    out = {}
    for dossier, _sous, fichiers in os.walk(racine):
        if any(fnmatch.fnmatch(f.lower(), motif.lower()) for f in fichiers):
            nom = nom_acquisition(dossier)
            if nom in out:
                raise SystemExit(f"deux acquisitions portent le nom {nom} :\n"
                                 f"  {out[nom]}\n  {dossier}")
            out[nom] = dossier
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("racine", help="dossier qui contient les acquisitions")
    ap.add_argument("--sortie", default="sorties")
    ap.add_argument("--motif", default=config.MOTIF_IMAGES)
    ap.add_argument("--coupe", action="store_true")
    ap.add_argument("--sur-cpu", action="store_true")
    ap.add_argument("--sans-figure", action="store_true")
    ap.add_argument("--modele-detection", default=None)
    ap.add_argument("--modele-maturite", default=None)
    ap.add_argument("--un", default=None, help="un seul test, par son nom")
    a = ap.parse_args()

    tous = acquisitions(a.racine, a.motif)
    noms = [a.un] if a.un else list(tous)
    os.makedirs(a.sortie, exist_ok=True)
    journal = os.path.join(a.sortie, "journal_lot.txt")
    print(f"{len(noms)} acquisitions a traiter")

    t0 = time.time()
    for i, nom in enumerate(noms, 1):
        sortie = os.path.join(a.sortie, nom)
        if os.path.exists(os.path.join(sortie, "resume.json")):
            continue
        t = time.time()
        try:
            r = traite(tous[nom], nom=nom, sortie=sortie, gpu=not a.sur_cpu,
                       coupe=a.coupe, modele_detection=a.modele_detection,
                       modele_maturite=a.modele_maturite, motif=a.motif,
                       scattergramme=not a.sans_figure, verbose=False)["resultats"]
            msg = (f"[{i}/{len(noms)}] {nom:32s} %RET {r['pct_ret']:6.2f}  "
                   f"IRF {r['irf']:6.2f}  RET {r['n_ret']:5d}  {time.time() - t:5.1f} s")
        except BaseException as e:                         # SystemExit compris
            if isinstance(e, KeyboardInterrupt):
                raise
            msg = f"[{i}/{len(noms)}] {nom:32s} ECHEC {type(e).__name__}: {e}"
            traceback.print_exc()
        print(msg, flush=True)
        with open(journal, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # ---- un tableau de tous les resultats disponibles ----------------------
    lignes = []
    for nom in sorted(os.listdir(a.sortie)):
        p = os.path.join(a.sortie, nom, "resume.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                r = json.load(f)["resultats"]
            lignes.append(dict(acquisition=nom, pct_ret=r["pct_ret"], irf=r["irf"],
                               n_rouges=r["n_rouges"], n_ret=r["n_ret"],
                               n_immatures=r["n_immatures"]))
    if lignes:
        with open(os.path.join(a.sortie, "resultats_lot.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter=";")
            w.writeheader()
            w.writerows(lignes)
    print(f"termine en {(time.time() - t0) / 60:.1f} min, "
          f"{len(lignes)} resultats dans {a.sortie}")


if __name__ == "__main__":
    main()
