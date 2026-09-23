"""
Ou sont les modeles, et les quelques constantes qui engagent le resultat.
========================================================================

**Les modeles ne sont pas dans ce depot.** Ils sont cherches, dans cet ordre :

  1. l'argument de ligne de commande (`--modele-detection`, `--modele-maturite`) ;
  2. les variables d'environnement `RET_MODELE_DETECTION`, `RET_MODELE_MATURITE` ;
  3. le dossier `models/` a la racine du depot, sous leurs noms de livraison.

Voir `models/README.md` pour ce que chaque fichier doit contenir.
"""
from __future__ import annotations

import os

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOSSIER_MODELES = os.path.join(RACINE, "models")

# Le nom du fichier du detecteur est FONCTIONNEL : Vision Hub lit l'option de
# decoupe (`d`) et le jeton `RET` dans ce nom (voir `edd4.parse_model_params`).
NOM_DETECTION = "EDD4d_RET-V3.3.000.onnx"
NOM_MATURITE = "modele_rapide_t8.pt"

# La porte de l'IRF, pour LE modele de maturite livre (`modele_rapide_t8.pt`).
# Le seul parametre ajuste de toute la chaine : pose une fois sur 25 echantillons
# d'apprentissage, juge sur 11 reserves (erreur absolue 2,39 pt, biais +0,13,
# pente de Passing-Bablok 0,937 [0,754 ; 1,173]). Un autre checkpoint a besoin
# de SA porte : celle du modele a 15 canaux valait 241,7403.
PORTE_IRF = 266.2375

# Les images d'une acquisition Vision : Picture01.jpg ... Picture60.jpg. Le
# dossier brut de l'automate contient aussi des FocusEdge_*.jpg, a ne pas lire.
MOTIF_IMAGES = "Picture*.jpg"


def modele_detection() -> str:
    return os.environ.get("RET_MODELE_DETECTION",
                          os.path.join(DOSSIER_MODELES, NOM_DETECTION))


def modele_maturite() -> str:
    return os.environ.get("RET_MODELE_MATURITE",
                          os.path.join(DOSSIER_MODELES, NOM_MATURITE))


def verifie_modele(chemin: str, role: str) -> None:
    if not os.path.isfile(chemin):
        raise SystemExit(
            f"modele de {role} introuvable : {chemin}\n"
            f"  les modeles ne sont pas distribues avec ce depot : voir "
            f"models/README.md, ou passer --modele-{role} / "
            f"RET_MODELE_{role.upper()}")
