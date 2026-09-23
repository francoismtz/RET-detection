# Modèles (non distribués)

Les deux modèles entraînés **ne sont pas versionnés dans ce dépôt**. La chaîne les
cherche ici sous leur nom de livraison, ou à l'endroit indiqué par
`--modele-detection` / `--modele-maturite`, ou par les variables d'environnement
`RET_MODELE_DETECTION` / `RET_MODELE_MATURITE`.

| Fichier | Rôle | Taille | SHA-256 |
|---|---|---|---|
| `EDD4d_RET-V3.3.000.onnx` | détecteur, 4 classes | 72,3 Mo | `0203de5a9c11c2781e08d52adda7027a6855dbce7a5cf2097fa7465f6834a2fa` |
| `modele_rapide_t8.pt` | U-Net de maturité | 1,5 Mo | `6ad8b1d79b7cf525cd22056101a01777a548f5c7a35a120f54e873119f75cb07` |

## Détecteur : `EDD4d_RET-V3.3.000.onnx`

- EfficientDet-D4 (`ssd_efficientnet-b4_bifpn_keras`), exporté depuis l'API
  TensorFlow Object Detection. ONNX opset 13.
- **Entrée** : `input_tensor`, `uint8`, NHWC `[1, H, W, 3]`, pixels bruts. Le
  redimensionnement en 1024 × 1024 est dans le graphe.
- **Sorties utilisées** : `detection_boxes` (normalisées, ordre
  `ymin, xmin, ymax, xmax`), `detection_classes` (1-basé), `detection_scores`,
  `detection_multiclass_scores`, `detection_anchor_indices`.
- **Classes** : `1 Debris`, `2 Reticulocyte`, `3 WBC`, `4 RBC`.
- **Le nom du fichier est fonctionnel** : Vision Hub y lit l'option de découpe
  (`d` dans `EDD4d`) et le jeton `RET`. Pour renommer le fichier, garder le
  préfixe `EDD4d_` et le jeton `RET`.
- Avec `--coupe`, la chaîne écrit une fois deux ONNX dérivés (tronc et queue)
  dans `coupe_gpu/`, à côté du modèle. Leur nom contient l'empreinte du modèle
  source.

Le modèle en service avant ce travail, `EDD4d_RET-V2.2.106.onnx`, suit le même
contrat et peut être passé à la place avec `--modele-detection`.

## U-Net de maturité : `modele_rapide_t8.pt`

Un dictionnaire enregistré avec `torch.save` :

| Clé | Contenu |
|---|---|
| `params` | architecture et normalisation (`archi="unet3"`, `filtres=16`, `taille=96`, `norm_lame="mediane"`…) |
| `noms_canaux` | `["z_nmb", "z_tot", "z_cos", "rang_nmb", "t_r", "t_g", "t_b", "cellule"]` |
| `seuil` | `0.5` |
| `poids` | liste de 3 `state_dict`, un par graine, moyennés à l'inférence |

`ret_inference/rapide.py` reconstruit l'architecture à partir de ces clés.
**La porte de l'IRF (`config.PORTE_IRF = 266.2375`) est propre à ce
checkpoint** : un autre modèle demande sa propre porte, ajustée sur des
échantillons d'apprentissage.
