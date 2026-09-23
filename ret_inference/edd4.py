"""
Inference ONNX du detecteur EDD4d_RET -- pre- et post-traitement.
=================================================================

**Il y a un post-traitement, et il n'est pas optionnel.** Donner une image au
graphe et lire ``detection_boxes`` rend des nombres qui ne veulent rien dire tant
que quatre conventions ne sont pas appliquees. Toutes les quatre sont celles du
code qui fait tourner ce modele en production, et ont ete verifiees cellule par
cellule contre le ``Recognition.xml`` de l'automate (422/422 et 412/412 cellules
sur deux acquisitions, geometrie au pixel pres, scores a 1e-5 pres).

1. **L'image est coupee avant l'inference.** Le ``d`` de ``EDD4d`` signifie
   *decoupe verticale* : une acquisition 4096x2160 est coupee en deux moities
   2048x2160, passees chacune dans le graphe, et les detections de la moitie
   droite sont decalees de +2048. Donner l'image entiere produirait en silence
   d'autres detections (moins bonnes) : un SSD decode ses boites contre l'entree
   qu'il a recue.

2. **Les boites sont normalisees ``[ymin, xmin, ymax, xmax]``** -- l'ordre de
   l'API TensorFlow Object Detection, ni le ``xywh`` de COCO, ni des pixels.

3. **Les classes sont des indices 1-bases** : ``1=Debris 2=Reticulocyte 3=WBC
   4=RBC`` (``label_map.pbtxt`` du modele).

4. **Un seuil de score et la resolution des doublons.** L'automate tourne a
   ``Algo_Threshold_Score=20`` (score >= 0,20). Le NMS est dans le graphe mais
   il est **par classe** : une meme cellule peut revenir deux ou trois fois sous
   des classes differentes (une hematie a 0,82 et la meme boite en reticulocyte
   a 0,22). On n'en garde qu'une, par score decroissant puis aire croissante,
   sur l'acquisition entiere (les deux moities ensemble).

Deux details different du portage de Vision Hub :

* l'automate **arrondit** les dimensions et l'origine de la boite ; ``int()``
  tronque. L'arrondi reproduit toutes les cellules de reference, la troncature
  ~23 %. L'ecart vaut au plus 1 px ;
* l'automate **supprime** un doublon (garde la meilleure classe) ; Vision Hub le
  *re-etiquette* ``Collision``. La suppression reproduit exactement le compte de
  l'automate. Les deux sont disponibles, voir ``DuplicatePolicy``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Sequence

import cv2
import numpy as np
import onnxruntime as ort

# Indice de sortie du graphe (1-base) -> nom canonique. C'est le `label_map.pbtxt`
# du modele, et la meme table pour V2.2.106 (en service) et V3.3.000 (retenu).
# Attention : ce ne sont PAS les `category_id` du COCO exporte par Vision Hub
# (12 Debris, 14 Reticulocyte, 16 RBC, 20 WBC). Lire avec la mauvaise table
# re-etiquette toutes les boites sans lever la moindre erreur.
MODEL_INDEX_TO_CLASS: dict[int, str] = {
    1: "Debris",
    2: "Reticulocyte",
    3: "WBC",
    4: "RBC",
    5: "Stuck cell",    # seulement pour un modele a 5 classes ; inerte ici
}

# Le reglage de l'automate, lu dans <Param Algo_Threshold_Score="20" />.
DEFAULT_MIN_SCORE = 20.0        # pourcent
DEFAULT_MIN_SCORE_DELTA = 0.0   # pourcent ; 0 desactive le declassement CheckScore
DUPLICATE_IOU = 0.5

# Que faire d'une seconde detection d'une cellule deja comptee.
#   "remove"   -- la supprimer, garder la meilleure classe (comme l'automate) ;
#   "relabel"  -- la marquer "Collision" et la garder (comme Vision Hub) ;
#   "keep"     -- tout garder (diagnostic seulement).
DuplicatePolicy = Literal["remove", "relabel", "keep"]

# Geometrie d'une acquisition.
FULL_WIDTH, FULL_HEIGHT = 4096, 2160


@dataclass
class Detection:
    """Une cellule predite, dans le repere de l'image entiere."""

    label: str          # nom canonique, ou "Collision"/"Unclassified"
    score: float
    center_x: int
    center_y: int
    width: int
    height: int
    multiclass: tuple[float, ...] = field(default=())
    raw_label: str = ""   # classe avant resolution des doublons / CheckScore
    anchor: int = -1      # indice d'ancre du graphe ; deux classes peuvent la partager
    quadrant: int = 0     # le quadrant dont elle a ete decodee

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` COCO, avec l'arithmetique entiere de l'automate."""
        return (
            int(self.center_x - self.width / 2),
            int(self.center_y - self.height / 2),
            self.width,
            self.height,
        )

    @property
    def area(self) -> int:
        return self.width * self.height


def load_session(
    model_path: str,
    threads: int | None = None,
    providers: Sequence[str] | None = None,
) -> ort.InferenceSession:
    """
    Ouvre le graphe. **Le defaut reste le CPU, et c'est un choix** : un noyau
    CUDA ne somme pas dans le meme ordre qu'un noyau CPU, les scores bougent dans
    leurs derniers chiffres, et cette chaine seuille a ``score >= 0,20`` et
    classe les doublons par score -- deux endroits ou un dernier chiffre peut
    changer le sort d'une cellule. Le GPU se demande explicitement.

    **Deux replis silencieux sur le CPU sont fermes ici**, a deux moments
    differents :

    1. *a la creation de la session*, quand les DLL d'un provider manquent : la
       session est construite sur CPU et ``get_providers()`` le dit, si on le
       lui demande -- on le lui demande ;
    2. *a la premiere inference*, quand un noyau refuse de s'initialiser :
       onnxruntime se reconstruit sur CPU **sur place**, apres avoir annonce
       CUDA. ``disable_fallback()`` transforme ce repli en exception.
    """
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    wanted = list(providers) if providers else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=opts,
                                   providers=wanted)
    got = session.get_providers()
    missing = [p for p in wanted if p not in got]
    if missing:
        raise RuntimeError(
            f"onnxruntime a laisse tomber {missing} en silence : il tourne sur "
            f"{got}. Providers de ce build : {ort.get_available_providers()}. "
            f"Un provider GPU dont les DLL manquent ne leve aucune exception : "
            f"on la leve ici plutot que de mesurer un GPU mysterieusement lent."
        )
    if wanted != ["CPUExecutionProvider"]:
        session.disable_fallback()
    return session


def read_image_rgb(path: str) -> np.ndarray:
    """Decode une image en RGB uint8. ``np.fromfile`` plutot que ``cv2.imread``
    pour que les chemins accentues passent sous Windows."""
    buf = np.fromfile(path, np.uint8)
    if buf.size == 0:
        raise OSError(f"fichier vide ou illisible : {path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"image non decodable : {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU de deux boites ``(x, y, w, h)``."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = float((x2 - x1) * (y2 - y1))
    union = float(a[2] * a[3]) + float(b[2] * b[3]) - inter
    return inter / union if union else 0.0


def run_quadrant(
    session,
    rgb: np.ndarray,
    x_offset: int = 0,
    y_offset: int = 0,
    min_score: float = DEFAULT_MIN_SCORE,
    min_score_delta: float = DEFAULT_MIN_SCORE_DELTA,
    quadrant: int = 0,
) -> list[Detection]:
    """
    Passe le graphe sur un quadrant et decode ses detections dans le repere de
    l'image parente.

    L'arrondi est celui de l'automate, etabli en reproduisant toutes les
    cellules de deux acquisitions de reference :

        w  = round((xmax - xmin) * W)      h  = round((ymax - ymin) * H)
        cx = round(xmin * W) + w // 2      cy = round(ymin * H) + h // 2

    Les *dimensions* et l'*origine* sont arrondies, mais le centre est ensuite
    l'origine plus une demi-largeur plancher, et non le vrai centre arrondi.
    Le vrai centre ne reproduit que ~49 % des cellules de reference ; ceci 100 %.
    """
    h, w = rgb.shape[:2]
    inp = rgb[None]  # NHWC, lot de 1 ; le redimensionnement est dans le graphe
    if inp.dtype != np.uint8:
        inp = inp.astype(np.uint8)

    names = [o.name for o in session.get_outputs()]
    outs = dict(zip(names, session.run(None, {session.get_inputs()[0].name: inp})))

    if not {"detection_boxes", "detection_classes", "detection_scores"} <= outs.keys():
        raise RuntimeError(
            f"le modele n'expose pas les sorties post-NMS ; recu {sorted(outs)}"
        )

    boxes = outs["detection_boxes"].reshape(-1, 4)     # normalisees ymin,xmin,ymax,xmax
    labels = outs["detection_classes"].flatten()
    scores = outs["detection_scores"].flatten()
    anchors = outs.get("detection_anchor_indices")
    anchors = anchors.flatten() if anchors is not None else None
    multi = outs.get("detection_multiclass_scores")
    multi = multi.reshape(len(labels), -1) if multi is not None else None

    dets: list[Detection] = []
    for i in range(len(labels)):
        if scores[i] * 100.0 < min_score:
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        rect_w = round((xmax - xmin) * w)
        rect_h = round((ymax - ymin) * h)
        index = int(labels[i])
        if index not in MODEL_INDEX_TO_CLASS:
            # nomme, pas avale : un defaut silencieux re-etiquetterait toutes les
            # boites d'une classe inconnue en quelque chose de plausible
            raise ValueError(
                f"le graphe a emis l'indice de classe {index}, absent de "
                f"MODEL_INDEX_TO_CLASS ({sorted(MODEL_INDEX_TO_CLASS)}). Pour un "
                "modele reentraine, l'ajouter depuis SON label_map.pbtxt."
            )
        det = Detection(
            label=MODEL_INDEX_TO_CLASS[index],
            score=float(scores[i]),
            center_x=round(xmin * w) + rect_w // 2 + x_offset,
            center_y=round(ymin * h) + rect_h // 2 + y_offset,
            width=rect_w,
            height=rect_h,
            multiclass=tuple(float(v) for v in multi[i]) if multi is not None else (),
            anchor=int(anchors[i]) if anchors is not None else -1,
            quadrant=quadrant,
        )
        det.raw_label = det.label
        dets.append(det)

    if min_score_delta > 0:
        for det in dets:
            _check_score(det, min_score_delta)
    return dets


def _check_score(det: Detection, min_score_delta: float) -> None:
    """Declasse en ``Unclassified`` quand les deux meilleures classes sont proches."""
    if len(det.multiclass) < 2:
        return
    top, second = sorted(det.multiclass, reverse=True)[:2]
    if (top - second) * 100.0 < min_score_delta:
        det.label = "Unclassified"


def resolve_duplicates(
    dets: list[Detection],
    policy: DuplicatePolicy = "remove",
    threshold: float = DUPLICATE_IOU,
) -> list[Detection]:
    """
    Ramene les detections par classe du graphe a une detection par cellule.

    Classement par score decroissant puis aire croissante -- la priorite de
    l'automate. Une detection est le doublon d'une precedente quand elle

    * **partage son ancre** : la meme boite, rendue une seconde fois sous une
      autre classe parce que le NMS tourne par classe ; ou
    * **la recouvre au-dela de** ``threshold`` : une autre ancre tombee sur la
      meme cellule. Sur les acquisitions de reference ces doublons sont a IoU
      0,87-0,96, loin de toute coupure plausible : le seuil n'est pas regle.

    Rend les detections survivantes, par score decroissant.
    """
    dets.sort(key=lambda d: (-d.score, d.area))
    if policy == "keep":
        return dets

    kept: list[Detection] = []
    kept_rects: list[tuple[int, int, int, int]] = []
    seen_anchors: set[tuple[int, int]] = set()
    duplicates: list[Detection] = []

    for det in dets:
        key = (det.quadrant, det.anchor)
        is_dup = det.anchor >= 0 and key in seen_anchors
        if not is_dup:
            rect = det.bbox
            is_dup = any(
                iou_xywh(rect, other) > threshold for other in kept_rects
            )
        if is_dup:
            duplicates.append(det)
            continue
        if det.anchor >= 0:
            seen_anchors.add(key)
        kept.append(det)
        kept_rects.append(det.bbox)

    if policy == "relabel":
        for det in duplicates:
            det.label = "Collision"
        merged = kept + duplicates
        merged.sort(key=lambda d: (-d.score, d.area))
        return merged
    return kept


def predict_full_image(
    session,
    path: str,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    min_score_delta: float = DEFAULT_MIN_SCORE_DELTA,
    duplicates: DuplicatePolicy = "remove",
    vertical_split: int = 1,
    horizontal_split: int = 0,
) -> list[Detection]:
    """
    Traite une acquisition entiere exactement comme l'automate : la couper en
    quadrants, passer chacun, puis resoudre les doublons globalement. Les
    coordonnees sont rendues dans le repere de l'image entiere.
    """
    rgb = read_image_rgb(path)
    height, width = rgb.shape[:2]
    q_h = height // (horizontal_split + 1)
    q_w = width // (vertical_split + 1)

    dets: list[Detection] = []
    quadrant = 0
    for i in range(vertical_split + 1):
        for j in range(horizontal_split + 1):
            quadrant += 1
            x0, y0 = i * q_w, j * q_h
            crop = rgb[y0:y0 + q_h, x0:x0 + q_w]
            dets.extend(run_quadrant(
                session, crop, x_offset=x0, y_offset=y0,
                min_score=min_score, min_score_delta=min_score_delta,
                quadrant=quadrant,
            ))
    return resolve_duplicates(dets, policy=duplicates)


def model_name(model_path: str) -> str:
    return os.path.splitext(os.path.basename(model_path))[0]


def parse_model_params(model_path: str) -> tuple[int, int, bool]:
    """
    ``(vertical_split, horizontal_split, resize_needed)`` lus dans le nom du
    fichier, comme le fait Vision Hub. ``EDD4d_RET-V3.3.000`` -> (1, 0, False).
    **Le nom du fichier est donc fonctionnel** : renommer le modele sans garder
    le prefixe ``EDD4d_`` casserait la decoupe dans Vision Hub.
    """
    name = model_name(model_path)
    head = name.split("_")[0]
    options = head[4:] if len(head) > 4 else ""
    vertical = 1 if ("d" in options or "quaters" in name) else 0
    horizontal = 1 if "quaters" in name else 0
    return vertical, horizontal, "r" in options
