"""
ret_inference -- detection, classification et maturite des reticulocytes.
=========================================================================

Une acquisition Vision (60 images 4096x2160, methode RET, coloration au
nouveau bleu de methylene) entre ; trois choses sortent :

  * le `Preclassifier_output.xml` au format Vision Hub, augmente du %RET et de
    la fraction de reticulocytes immatures (IRF) ;
  * le scattergramme RET, a la facon du Sysmex XN ;
  * un `.npz` avec les grandeurs de chaque globule rouge.

    from ret_inference.chaine import traite
    resultats = traite("acquisition/pictures", sortie="sorties/acquisition")

Modules, dans l'ordre de la chaine :

    edd4.py         inference ONNX du detecteur EDD4 et son post-traitement
    coupe_gpu.py    (optionnel) le detecteur coupe : tronc GPU, NMS CPU
    detection.py    passe EDD4 sur l'acquisition + selection des globules rouges
    noyau.py        physique : densite optique, deconvolution, masque de cellule
    reseau.py       le U-Net
    rapide.py       l'extracteur en lot (entrees calculees sur GPU)
    maturite.py     loi de lame, puis mesure de `douce` sur chaque cellule
    recognition.py  %RET, IRF et le XML Vision Hub
    figures.py      le scattergramme
    chaine.py       l'orchestrateur et la ligne de commande
"""

__version__ = "1.0.0"
