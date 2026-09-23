"""
La chaine complete, d'une acquisition a son XML -- tout sur le GPU.
===================================================================

    python -m ret_inference chemin/vers/acquisition/pictures --sortie sorties/

Quatre etapes, dans cet ordre, dans un seul processus :

  1. **detection**  EDD4d_RET-V3.3.000 sur la CUDA EP d'onnxruntime, image par
     image, avec le post-traitement de l'automate (`edd4.py`), puis la
     selection des globules rouges (`detection.py`) ;
  2. **loi de lame** la mediane par canal de 400 hematies de l'acquisition,
     tirees une image sur cinq. C'est la seule calibration de la chaine ;
  3. **maturite**   le U-Net rapide (8 canaux) sur TOUS les globules rouges
     retenus, trois reseaux moyennes sur le GPU. Il rend `douce`, la somme de
     la probabilite de reticulum sur la cellule ;
  4. **sorties**    le `Preclassifier_output.xml` au format Vision Hub, augmente
     du %RET et de l'IRF, le scattergramme RET, les grandeurs par cellule.

Ce qui est mesure et ce qui ne l'est pas :

  * le **%RET** est un rapport de comptes du detecteur. Il ne se calibre pas ;
  * l'**IRF** est la part des reticulocytes dont `douce` depasse la porte
    (`config.PORTE_IRF`), posee une fois sur 25 echantillons et jugee sur 11
    reserves. Elle est en dur, et c'est un choix defendu.

`--sur-cpu` refait la meme chaine sans le GPU : c'est le temoin.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import config


def nom_acquisition(dossier_images):
    """Le nom d'une acquisition : son dossier, ou le parent d'un `pictures/`."""
    d = os.path.normpath(os.path.abspath(dossier_images))
    base = os.path.basename(d)
    return os.path.basename(os.path.dirname(d)) if base.lower() == "pictures" else base


def traite(dossier_images, nom=None, sortie=None, gpu=True, coupe=False,
           modele_detection=None, modele_maturite=None, motif=config.MOTIF_IMAGES,
           scattergramme=True, dpi=150, dossier_analyse=None, verbose=True):
    """-> le dictionnaire de resultats, chronos compris."""
    from . import detection as D
    from . import figures as F
    from . import maturite as M
    from . import recognition as R
    from .edd4 import model_name

    if not os.path.isdir(dossier_images):
        raise SystemExit(f"dossier d'images introuvable : {dossier_images}")
    modele_detection = modele_detection or config.modele_detection()
    modele_maturite = modele_maturite or config.modele_maturite()
    config.verifie_modele(modele_detection, "detection")
    config.verifie_modele(modele_maturite, "maturite")
    nom = nom or nom_acquisition(dossier_images)
    sortie = sortie or os.path.join("sorties", nom)
    os.makedirs(sortie, exist_ok=True)
    appareil = "auto" if gpu else "cpu"

    barre = "=" * 78
    if verbose:
        print(barre)
        print(f"CHAINE COMPLETE -- {nom}   ({'GPU' if gpu else 'CPU'})")
        print(barre)

    t_total = time.time()

    # ---- 1. detection ----------------------------------------------------
    if verbose:
        print("\n[1/4] DETECTION")
    cel, c_det = D.detecte(dossier_images, nom, modele=modele_detection,
                           gpu=gpu, coupe=coupe, motif=motif, verbose=verbose)

    # ---- 2. loi de lame --------------------------------------------------
    if verbose:
        print("\n[2/4] LOI DE LAME")
    M.charge_reseau(modele_maturite, verbose=verbose, appareil=appareil)
    med, c_lame = M.loi_de_lame(cel, dossier_images, verbose=verbose)

    # ---- 3. maturite -----------------------------------------------------
    if verbose:
        print("\n[3/4] MATURITE -- le U-Net sur tous les globules rouges")
    grandeurs, c_mat = M.mesure(cel, dossier_images, med, modele=modele_maturite,
                                verbose=verbose, appareil=appareil)

    # ---- 4. sorties ------------------------------------------------------
    if verbose:
        print("\n[4/4] SORTIES")
    t0 = time.time()
    res = R.resume(cel, grandeurs)
    # Vision Hub reconnait un dossier d'analyse au `Preclassifier_output.xml`
    # qu'il contient, jamais a son nom : le nom n'est qu'un libelle d'affichage.
    dossier_analyse = dossier_analyse or (
        f"ret_inference_{model_name(modele_detection)}_No_{len(cel.fichiers)}")
    xml = R.ecrit(os.path.join(sortie, dossier_analyse, R.NOM_FICHIER),
                  cel, grandeurs, res, modele=model_name(modele_detection),
                  modele_maturite=os.path.basename(modele_maturite))
    png = None
    if scattergramme:
        png = F.scattergramme(cel, grandeurs, res,
                              os.path.join(sortie, "scattergramme_RET.png"),
                              dpi=dpi)
    np.savez_compressed(
        os.path.join(sortie, "cellules.npz"),
        image=cel.image, classe=cel.classe, score=cel.score,
        x0=cel.x0, y0=cel.y0, w=cel.w, h=cel.h, cx=cel.cx, cy=cel.cy,
        couture=cel.couture, **grandeurs)
    t_sorties = time.time() - t0
    total = time.time() - t_total

    chrono = dict(
        acquisition=nom, gpu=gpu, n_images=len(cel.fichiers),
        **c_det, **c_lame, **c_mat,
        t_sorties=t_sorties, t_total=total,
        part_detection=100 * c_det["t_detection"] / total,
        part_loi_lame=100 * c_lame["t_loi_lame"] / total,
        part_maturite=100 * c_mat["t_maturite"] / total,
    )
    tout = dict(resultats=res, chronos=chrono,
                fichiers=dict(xml=xml, scattergramme=png))
    with open(os.path.join(sortie, "resume.json"), "w", encoding="utf-8") as f:
        json.dump(tout, f, indent=1, ensure_ascii=False, default=float)

    if verbose:
        print(f"  {R.NOM_FICHIER} : {os.path.getsize(xml) / 1e6:.1f} Mo"
              f"   ({dossier_analyse}/)")
        if png:
            print(f"  scattergramme   : {os.path.basename(png)}")
        print(f"  ecriture        : {t_sorties:.1f} s")
        print("\n" + barre)
        print("RESULTATS")
        print(barre)
        print(f"  cellules detectees          {res['n_detectees']:>8}")
        print(f"  globules rouges retenus     {res['n_rouges']:>8}   "
              f"({res['n_rbc']} RBC + {res['n_ret']} reticulocytes)")
        print(f"  %RET                        {res['pct_ret']:>8.3f} %")
        print(f"  IRF   (porte {res['porte_irf']:.1f})     "
              f"{res['irf']:>8.3f} %   "
              f"({res['n_immatures']} / {res['n_ret_mesures']})")
        print("\n" + barre)
        print(f"TEMPS -- un test de {len(cel.fichiers)} images, "
              f"{'GPU' if gpu else 'CPU'}")
        print(barre)
        for etiquette, cle in (("detection EDD4", "t_detection"),
                               ("loi de lame", "t_loi_lame"),
                               ("maturite U-Net", "t_maturite"),
                               ("XML + figures", "t_sorties")):
            print(f"  {etiquette:22s}{chrono[cle]:8.1f} s"
                  f"{100 * chrono[cle] / total:8.1f} %")
        print(f"  {'chargement detecteur':22s}{chrono['t_charge']:8.1f} s")
        print(f"  {'-' * 40}")
        print(f"  {'TOTAL':22s}{total:8.1f} s   = {total / 60:.1f} min")
        print(f"\n  {chrono['t_par_image']:.2f} s/image en detection, "
              f"{chrono['ms_par_cellule']:.2f} ms/cellule en maturite")
        print(f"  -> {os.path.abspath(sortie)}")
    return tout


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m ret_inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="dossier des images d'une acquisition "
                                   "(Picture01.jpg ... Picture60.jpg)")
    ap.add_argument("--nom", default=None,
                    help="nom de l'acquisition (defaut : le nom du dossier)")
    ap.add_argument("--sortie", default=None,
                    help="dossier de sortie (defaut : sorties/<nom>)")
    ap.add_argument("--modele-detection", default=None,
                    help=f"ONNX du detecteur (defaut : models/{config.NOM_DETECTION})")
    ap.add_argument("--modele-maturite", default=None,
                    help=f"checkpoint du U-Net (defaut : models/{config.NOM_MATURITE})")
    ap.add_argument("--motif", default=config.MOTIF_IMAGES,
                    help="motif des images a lire (defaut : %(default)s)")
    ap.add_argument("--coupe", action="store_true",
                    help="detecteur en graphe coupe : ~1,5x plus rapide, memes "
                         "cellules (demande le paquet `onnx`)")
    ap.add_argument("--sur-cpu", action="store_true",
                    help="temoin : la meme chaine sans le GPU")
    ap.add_argument("--sans-figure", action="store_true",
                    help="ne pas tracer le scattergramme")
    ap.add_argument("--dossier-analyse", default=None,
                    help="nom du dossier d'analyse lu par Vision Hub")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)

    traite(a.images, nom=a.nom, sortie=a.sortie, gpu=not a.sur_cpu,
           coupe=a.coupe, modele_detection=a.modele_detection,
           modele_maturite=a.modele_maturite, motif=a.motif,
           scattergramme=not a.sans_figure, dpi=a.dpi,
           dossier_analyse=a.dossier_analyse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
