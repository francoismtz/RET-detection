"""
EDD4 sur GPU : couper le graphe, pas le modifier (option ``--coupe``).
=====================================================================

**La V3.3.000 n'a PAS besoin de ce module.** Le detecteur tourne tel quel sur la
CUDA EP, treize fois plus vite que sur CPU. Ce fichier n'est qu'une
optimisation supplementaire (0,16 s par image au lieu de 0,25), pour des
cellules rigoureusement identiques a la voie directe.

Il a ete ecrit pour l'ancien detecteur, **V2.2.106**, ou c'etait le seul moyen :
son noyau CUDA ``NonMaxSuppression`` abat le processus (``CUDA failure 700: an
illegal memory access``), reproduit sur trois versions d'onnxruntime-gpu et deux
de cuDNN. Or le NMS ne coute rien : 6 % du temps sur CPU. Les 94 % restants sont
EfficientDet-D4, et **ce tronc va 9,3 fois plus vite sur la carte**.

D'ou ce module : **le tronc sur le GPU, la queue sur le CPU**. C'est un
placement, pas une reecriture -- les memes noeuds ONNX, dans le meme ordre, avec
les memes constantes. Le graphe est coupe sur les tenseurs qui separent les 1764
noeuds du tronc (dont toutes les convolutions) des 397 de la queue (les 4 NMS et
leur post-traitement). Rien n'est reimplemente.

    session = charge_coupee(MODELE)                 # tronc CUDA, queue CPU
    dets = predict_full_image(session, chemin)      # inchange

L'objet rendu expose ``get_inputs``, ``get_outputs``, ``get_providers`` et
``run`` : c'est tout ce que ``edd4`` demande d'une session.

Les deux ONNX derives sont ecrits une fois, a cote du modele, et nommes d'apres
l'empreinte SHA-256 du modele source : deux versions du detecteur ne peuvent pas
se marcher dessus, et un modele remplace invalide son cache tout seul.
"""
from __future__ import annotations

import hashlib
import os
from collections import namedtuple

import onnxruntime as ort

Entree = namedtuple("Entree", "name type shape")
Sortie = namedtuple("Sortie", "name")

TRONC_DEFAUT = ["CUDAExecutionProvider", "CPUExecutionProvider"]
QUEUE_DEFAUT = ["CPUExecutionProvider"]


def _partitionne(graphe):
    """-> (noeuds de la queue, tenseurs produits par la queue, frontiere).

    La queue = les NonMaxSuppression et tout ce qui en descend. La frontiere =
    ce que la queue lit et qu'elle ne produit pas.
    """
    consommateurs: dict[str, list] = {}
    for n in graphe.node:
        for i in n.input:
            consommateurs.setdefault(i, []).append(n)

    vus, pile = set(), [n for n in graphe.node if n.op_type == "NonMaxSuppression"]
    if not pile:
        raise RuntimeError("aucun NonMaxSuppression dans ce graphe : "
                           "rien a contourner, utilisez une session normale.")
    while pile:
        n = pile.pop()
        if id(n) in vus:
            continue
        vus.add(id(n))
        for o in n.output:
            pile.extend(consommateurs.get(o, []))

    queue = [n for n in graphe.node if id(n) in vus]
    produits_queue = {o for n in queue for o in n.output}
    initialiseurs = {i.name for i in graphe.initializer}
    frontiere = sorted({i for n in queue for i in n.input
                        if i and i not in produits_queue and i not in initialiseurs})
    return queue, produits_queue, frontiere


def construit(modele: str, dossier: str | None = None, refaire: bool = False):
    """Ecrit (une fois) les deux moities et rend leurs chemins + le contrat."""
    import onnx

    dossier = dossier or os.path.join(os.path.dirname(os.path.abspath(modele)),
                                      "coupe_gpu")
    os.makedirs(dossier, exist_ok=True)
    with open(modele, "rb") as f:
        empreinte = hashlib.sha256(f.read()).hexdigest()[:12]
    base = os.path.join(dossier, f"{os.path.splitext(os.path.basename(modele))[0]}"
                                 f"_{empreinte}")
    p_tronc, p_queue = base + "_tronc.onnx", base + "_queue.onnx"

    m = onnx.load(modele)
    g = m.graph
    entree = g.input[0].name
    noms_sorties = [o.name for o in g.output]
    _queue, produits_queue, frontiere = _partitionne(g)
    sorties_queue = [o for o in noms_sorties if o in produits_queue]
    sorties_tronc = [o for o in noms_sorties if o not in produits_queue]

    if refaire or not (os.path.exists(p_tronc) and os.path.exists(p_queue)):
        onnx.utils.extract_model(modele, p_tronc, [entree],
                                 sorties_tronc + frontiere)
        onnx.utils.extract_model(modele, p_queue, frontiere, sorties_queue)

    return dict(tronc=p_tronc, queue=p_queue, entree=entree,
                frontiere=frontiere, sorties=noms_sorties,
                sorties_tronc=sorties_tronc, sorties_queue=sorties_queue)


class SessionCoupee:
    """Deux sessions onnxruntime derriere l'interface d'une seule."""

    def __init__(self, contrat, s_tronc, s_queue):
        self._c = contrat
        self.tronc, self.queue = s_tronc, s_queue
        e = s_tronc.get_inputs()[0]
        self._entree = Entree(e.name, e.type, e.shape)
        self._sorties = [Sortie(n) for n in contrat["sorties"]]

    def get_inputs(self):
        return [self._entree]

    def get_outputs(self):
        return list(self._sorties)

    def get_providers(self):
        return list(dict.fromkeys(self.tronc.get_providers()
                                  + self.queue.get_providers()))

    def run(self, noms, entrees):
        c = self._c
        demandes = c["sorties"] if noms is None else list(noms)
        # le tronc rend ses sorties propres ET la frontiere, en une passe
        a_demander = [n for n in c["sorties_tronc"] if n in demandes] + c["frontiere"]
        vals = self.tronc.run(a_demander, entrees)
        table = dict(zip(a_demander, vals))
        besoin_queue = [n for n in demandes if n in c["sorties_queue"]]
        if besoin_queue:
            passage = {n: table[n] for n in c["frontiere"]}
            table.update(zip(besoin_queue, self.queue.run(besoin_queue, passage)))
        return [table[n] for n in demandes]


def charge_coupee(modele: str, providers_tronc=None, providers_queue=None,
                  threads: int | None = None, dossier: str | None = None):
    """La session coupee, prete a passer a ``edd4.predict_full_image``."""
    c = construit(modele, dossier)
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    voulus = list(providers_tronc or TRONC_DEFAUT)
    s_tronc = ort.InferenceSession(c["tronc"], sess_options=opts,
                                   providers=voulus)
    manquants = [p for p in voulus if p not in s_tronc.get_providers()]
    if manquants:
        raise RuntimeError(
            f"onnxruntime a laisse tomber {manquants} en silence : il tourne sur "
            f"{s_tronc.get_providers()}. Build : {ort.get_available_providers()}")
    if voulus != ["CPUExecutionProvider"]:
        s_tronc.disable_fallback()
    s_queue = ort.InferenceSession(c["queue"], sess_options=opts,
                                   providers=list(providers_queue or QUEUE_DEFAUT))
    return SessionCoupee(c, s_tronc, s_queue)
