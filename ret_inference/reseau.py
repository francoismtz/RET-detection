"""Le reseau d'extraction du reticulum, en PyTorch.

Le modele livre est un `unet3` : deux niveaux de sous-echantillonnage, 16
filtres au premier niveau, 32 au second, 64 au fond ; chaque niveau enchaine
deux convolutions 3x3 suivies d'une normalisation par lot et d'une ReLU ; la
remontee sur-echantillonne au plus proche voisin, concatene le saut
correspondant, et la sortie est une convolution 1x1 vers un logit. 8 canaux en
entree, 119 313 parametres, champ receptif ~37 px : de quoi voir une tache de
reticulum et son voisinage immediat, pas la cellule entiere -- c'est voulu, la
place du pixel dans la cellule lui est donnee par les canaux relatifs a la
cellule.

Les autres options (profondeur, normalisation par groupes, dropout, blocs
residuels, convolutions dilatees ou separables) servaient au plan d'experiences
et sont gardees pour que tout checkpoint de l'etude se recharge.
"""
import torch
import torch.nn as nn


def appareil(choix="auto"):
    if choix == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choix)


def _init_conv(c):
    nn.init.kaiming_normal_(c.weight, mode="fan_in", nonlinearity="relu")
    if c.bias is not None:
        nn.init.zeros_(c.bias)


def _glorot(c):
    nn.init.xavier_uniform_(c.weight)
    if c.bias is not None:
        nn.init.zeros_(c.bias)


def _norme(kind, f):
    if kind == "bn":
        return nn.BatchNorm2d(f, eps=1e-3, momentum=0.01)
    if kind == "gn":
        return nn.GroupNorm(min(8, f), f, eps=1e-3)
    if kind in (None, "none", ""):
        return None
    raise KeyError(kind)


def _conv(n_in, f, k=3, dilatation=1, separable=False, biais=True):
    pad = dilatation * (k // 2)
    if separable and k > 1:
        dw = nn.Conv2d(n_in, n_in, k, padding=pad, dilation=dilatation,
                       groups=n_in, bias=False)
        pw = nn.Conv2d(n_in, f, 1, bias=biais)
        _init_conv(dw)
        _init_conv(pw)
        return nn.Sequential(dw, pw)
    c = nn.Conv2d(n_in, f, k, padding=pad, dilation=dilatation, bias=biais)
    _init_conv(c)
    return c


class Bloc(nn.Module):
    """2 x (conv 3x3 -> norme -> ReLU), avec raccourci residuel optionnel."""

    def __init__(self, n_in, f, norm="bn", dropout=0.0, dilatation=1,
                 separable=False, residuel=False):
        super().__init__()
        couches = []
        for i in range(2):
            couches.append(_conv(n_in if i == 0 else f, f,
                                 dilatation=(dilatation if i == 1 else 1),
                                 separable=separable, biais=(norm != "bn")))
            n = _norme(norm, f)
            if n is not None:
                couches.append(n)
            couches.append(nn.ReLU(inplace=True))
        if dropout > 0:
            couches.append(nn.Dropout2d(dropout))
        self.seq = nn.Sequential(*couches)
        self.raccourci = None
        if residuel:
            self.raccourci = (nn.Identity() if n_in == f
                              else nn.Conv2d(n_in, f, 1, bias=False))
            if not isinstance(self.raccourci, nn.Identity):
                _init_conv(self.raccourci)

    def forward(self, x):
        y = self.seq(x)
        return y if self.raccourci is None else y + self.raccourci(x)


class UNet(nn.Module):
    """`unet2` : 1 pooling. `unet3` : 2. `unet1` : 0 pooling mais deux blocs.
    `fcn` : quatre blocs a pleine resolution. Sortie : logits Nx1xHxW."""

    def __init__(self, n_in, archi="unet2", filtres=16, norm="bn", dropout=0.0,
                 dilatation_fond=1, separable=False, residuel=False):
        super().__init__()
        f = self.filtres = filtres
        self.archi = archi
        opt = dict(norm=norm, dropout=dropout, separable=separable,
                   residuel=residuel)
        if archi == "fcn":
            self.corps = nn.Sequential(
                Bloc(n_in, f, **opt), Bloc(f, 2 * f, **opt),
                Bloc(2 * f, 2 * f, dilatation=dilatation_fond, **opt),
                Bloc(2 * f, f, **opt))
            self.sortie = nn.Conv2d(f, 1, 1)
            _glorot(self.sortie)
            return
        n_pool = {"unet1": 0, "unet2": 1, "unet3": 2}[archi]
        self.n_pool = n_pool
        self.descend = nn.ModuleList()
        entree = n_in
        for i in range(n_pool):
            self.descend.append(Bloc(entree, f * (2 ** i), **opt))
            entree = f * (2 ** i)
        self.fond = Bloc(entree, f * (2 ** n_pool),
                         dilatation=dilatation_fond, **opt)
        self.remonte = nn.ModuleList()
        haut = f * (2 ** n_pool)
        for i in reversed(range(n_pool)):
            self.remonte.append(Bloc(haut + f * (2 ** i), f * (2 ** i), **opt))
            haut = f * (2 ** i)
        self.sortie = nn.Conv2d(haut, 1, 1)
        _glorot(self.sortie)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        if self.archi == "fcn":
            return self.sortie(self.corps(x))
        skips = []
        for bloc in self.descend:
            x = bloc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.fond(x)
        for j, bloc in enumerate(self.remonte):
            x = self.up(x)
            x = torch.cat([x, skips[self.n_pool - 1 - j]], dim=1)
            x = bloc(x)
        return self.sortie(x)


def n_parametres(m):
    return sum(p.numel() for p in m.parameters())
