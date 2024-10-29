from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from fastcore.all import is_listy, store_attr
from torch import Tensor, nn

from ....utils.misc import hard_identity, to_device
from ..initialisations import lookup_normal_init
from ..layers.activations import lookup_act
from ..layers.batchnorms import LCBatchNorm1d
from ..layers.self_attention import SelfAttention

__all__ = ["GraphCollapser", "NodePredictor", "InteractionNet", "GravNet", "GravNetLayer"]


class AbsGraphBlock(nn.Module):
    r"""
    Abstract class for implementing graph neural-network layers and blocks

    Arguments:
        n_v: number of vertices per data point to expect
        n_fpv: number of features per vertex to expect
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d`
    """

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
    ):
        super().__init__()
        store_attr()

    def _get_nn(self, n_in: int, n_outs: List[int]) -> nn.Sequential:
        if not is_listy(n_outs):
            n_outs = [n_outs]
        return nn.Sequential(*[self._get_layer(n_in if i == 0 else n_outs[i - 1], n) for i, n in enumerate(n_outs)])

    def _get_layer(self, fan_in: int, fan_out: int) -> nn.Module:
        layers = [nn.Linear(fan_in, fan_out)]
        self.lookup_init(self.act, fan_in, fan_out)(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if self.act != "linear":
            layers.append(self.lookup_act(self.act))
        if self.bn:
            layers.append(LCBatchNorm1d(self.bn_class(fan_out)))
        if self.do:
            if self.act == "selu":
                layers.append(nn.AlphaDropout(self.do))
            else:
                layers.append(nn.Dropout(self.do))
        return nn.Sequential(*layers)


class AbsGraphFeatExtractor(AbsGraphBlock, metaclass=ABCMeta):
    r"""
    Abstract class for implementing feature extraction for graph neural-network blocks.
    Overridden `forward` should return features per vertex.

    Arguments:
        n_v: number of vertices per data point to expect
        n_fpv: number of features per vertex to expect
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d`
    """

    row_wise: Optional[bool] = None

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
    ):
        super().__init__(
            n_v=n_v,
            n_fpv=n_fpv,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )

    @abstractmethod
    def get_out_size(self) -> Tuple[int, int]:
        pass  # N-vertices, N-features per vertex


class GraphCollapser(AbsGraphBlock):
    r"""
    Class for collapsing features per vertex (batch x vertices x features) down to flat data (batch x features).
    Can act in two ways:
        1. Compute aggregate features by taking the average and maximum of each feature across all vertices (does not assume any order to the vertices)
        2. Flatten out the vertices by reshaping (does assume an ordering to the vertices)
    Regardless of flattening approach, features per vertex can be revised beforehand via neural networks and self-attention.

    Arguments:
        n_v: number of vertices per data point to expect
        n_fpv: number of features per vertex to expect
        flatten: if True will flatten (reshape) data into (batch x features), otherwise will compute aggregate features (average and max)
        f_initial_outs: list of widths for the NN layers in an NN before self-attention (None = no NN)
        n_sa_layers: number of self-attention layers (outputs will be fed into subsequent layers)
        sa_width: width of self attention representation (paper recommends n_fpv//4)
        f_final_outs: list of widths for the NN layers in an NN after self-attention (None = no NN)
        global_feat_vec: if true and f_initial_outs or f_final_outs are not None,
            will concatenate the mean of each feature as new features to each vertex prior to the last network.
        agg_methods: list of text representations of aggregation methods. Default is mean and max.
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d`
        sa_class: class to use for self-attention layers, default is :class:`~lumin.nn.models.layers.self_attention.SelfAttention`
    """

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        flatten: bool,
        f_initial_outs: Optional[List[int]] = None,
        n_sa_layers: int = 0,
        sa_width: Optional[int] = None,
        f_final_outs: Optional[List[int]] = None,
        global_feat_vec: bool = False,
        agg_methods: Union[List[str], str] = ["mean", "max"],
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
        sa_class: Callable[[int], nn.Module] = SelfAttention,
    ):
        super().__init__(
            n_v=n_v,
            n_fpv=n_fpv,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )
        store_attr(but=["n_v", "n_fpv", "do", "bn", "act", "lookup_init", "lookup_act", "bn_class", "agg_methods"])
        self._check_agg_methods(agg_methods)

        self.gfv_pos = None
        if self.f_initial_outs is None:
            self.f_inital = hard_identity
            self.f_initial_outs = [self.n_fpv]
        else:
            if not is_listy(self.f_initial_outs):
                self.f_initial_outs = [self.f_initial_outs]
            fpv = 2 * self.n_fpv if self.global_feat_vec else self.n_fpv
            self.f_inital = self._get_nn(fpv, self.f_initial_outs)
            if self.global_feat_vec:
                self.gfv_pos = "pre-initial"

        if self.n_sa_layers > 0:
            if self.sa_width is None:
                raise ValueError("Please set a value for sa_width, the width of the self-attention layers.")
            self.sa_layers = nn.ModuleList(
                [
                    self.sa_class(
                        n_fpv=self.f_initial_outs[-1],
                        n_a=self.sa_width,
                        act=act,
                        bn=bn,
                        do=do,
                        lookup_act=lookup_act,
                        lookup_init=self.lookup_init,
                    )
                    for _ in range(self.n_sa_layers)
                ]
            )

        if self.f_final_outs is None:
            self.f_final = hard_identity
            self.f_final_outs = (
                [self.f_initial_outs[-1]] if self.n_sa_layers == 0 else [self.f_initial_outs[-1] * (self.n_sa_layers)]
            )
        else:
            if not is_listy(self.f_final_outs):
                self.f_final_outs = [self.f_final_outs]
            fpv = self.f_initial_outs[-1] if self.n_sa_layers == 0 else self.f_initial_outs[-1] * (self.n_sa_layers)
            if self.global_feat_vec and self.gfv_pos is None:
                fpv *= 2
            self.f_final = self._get_nn(fpv, self.f_final_outs)
            if self.global_feat_vec and self.gfv_pos is None:
                self.gfv_pos = "pre-final"

    @staticmethod
    def _agg_mean(x: Tensor) -> Tensor:
        return torch.mean(x, dim=1)

    @staticmethod
    def _agg_max(x: Tensor) -> Tensor:
        return torch.max(x, dim=1)[0]

    @staticmethod
    def _agg_absmax(x: Tensor) -> Tensor:
        return torch.max(x.abs(), dim=1)[0]

    def _check_agg_methods(self, agg_methods: Union[List[str], str]) -> None:
        self.agg_methods = []
        if not is_listy(agg_methods):
            agg_methods = [agg_methods]
        for m in agg_methods:
            m = m.lower()
            if m == "mean":
                self.agg_methods.append(self._agg_mean)
            elif m == "max":
                self.agg_methods.append(self._agg_max)
            elif m == "absmax":
                self.agg_methods.append(self._agg_absmax)
            else:
                raise ValueError(f"{m} not in [mean, max, absmax]")

    def _agg(self, x: Tensor) -> Tensor:
        if self.flatten:
            return x.reshape((len(x), -1))
        else:
            return torch.cat([agg(x) for agg in self.agg_methods], -1)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Collapses features per vertex down to features

        Arguments:
            x: incoming data (batch x vertices x features)

        Returns:
            Flattened data (batch x flat features)
        """

        if self.global_feat_vec and self.gfv_pos == "pre-initial":
            x = torch.cat(
                [x, x.mean(1).unsqueeze(2).repeat_interleave(repeats=x.shape[1], dim=2).transpose(1, 2)], dim=2
            )
        x = self.f_inital(x)
        if self.n_sa_layers > 0:
            outs = []
            for i, sa in enumerate(self.sa_layers):
                outs.append(sa(outs[-1] if i > 0 else x))
            x = torch.cat(outs, dim=-1)
        if self.global_feat_vec and self.gfv_pos == "pre-final":
            x = torch.cat(
                [x, x.mean(1).unsqueeze(2).repeat_interleave(repeats=x.shape[1], dim=2).transpose(1, 2)], dim=2
            )
        x = self.f_final(x)
        return self._agg(x)

    def get_out_size(self) -> int:
        r"""
        Get size of output

        Returns:
            Width of output representation
        """

        return self.f_final_outs[-1] * self.n_v if self.flatten else self.f_final_outs[-1] * len(self.agg_methods)


class NodePredictor(GraphCollapser):
    r"""
    Modified :class:`~lumin.nn.models.blocks.gnn_blocks.GraphCollapser` for providing a set of predictions per node in a graph,
    i.e collapsing features per vertex (batch x vertices x features) down to predictions per vertex, with approriate output activation functions data (batch x vertices x predictions).
    For compatibility with the format expected by some loss functions (e.g. `torch.nn.NLLLoss`), the output can be transposed to (batch x predictions x vertices).
    Features per vertex can be revised beforehand via neural networks and self-attention.
    The `f_final` neural network can be used to transform the input size to the required number of predicitons per node.

    .. Important::
        Since predictions are being provided in the `head` part of the model, but LUMIN expects models to also have a `body` and `tail` section.
        It is strongly recommended to use the :class:`~lumin.nn.models.blocks.body.IdentBody` and :class:`~lumin.nn.models.blocks.tail.IdentTail` modules to be placeholders.

    Arguments:
        n_v: number of vertices per data point to expect
        n_fpv: number of features per vertex to expect
        out_act: Output activation function to apply to every set of prediction per vertex. The weight initialisation of the last NN layer will be automatically set.
        transpose_out: If True, will transpose the putput into (batch x predictions x vertices), otherwise the output will be (batch x vertices x predictions)
        f_initial_outs: list of widths for the NN layers in an NN before self-attention (None = no NN)
        n_sa_layers: number of self-attention layers (outputs will be fed into subsequent layers)
        sa_width: width of self attention representation (paper recommends n_fpv//4)
        f_final_outs: list of widths for the NN layers in an NN after self-attention (None = no NN)
        global_feat_vec: if true and f_initial_outs or f_final_outs are not None,
            will concatenate the mean of each feature as new features to each vertex prior to the last network.
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d`
        sa_class: class to use for self-attention layers, default is :class:`~lumin.nn.models.layers.self_attention.SelfAttention`
    """

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        out_act: str,
        transpose_out: bool,
        f_initial_outs: Optional[List[int]] = None,
        n_sa_layers: int = 0,
        sa_width: Optional[int] = None,
        f_final_outs: Optional[List[int]] = None,
        global_feat_vec: bool = False,
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
        sa_class: Callable[[int], nn.Module] = SelfAttention,
    ):
        super().__init__(
            n_v=n_v,
            n_fpv=n_fpv,
            flatten=True,
            f_initial_outs=f_initial_outs,
            n_sa_layers=n_sa_layers,
            sa_width=sa_width,
            f_final_outs=f_final_outs,
            global_feat_vec=global_feat_vec,
            sa_class=sa_class,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )
        self.transpose_out = transpose_out
        if self.f_final_outs is not None and isinstance(self.f_final, nn.Module):
            self.lookup_init(out_act, self.f_final[-1][0].weight.shape[1], self.f_final[-1][0].weight[0])(
                self.f_final[-1][0].weight
            )
            self.f_final[-1] = self.f_final[-1][0]
        self.out_act = self.lookup_act(out_act)

    def _agg(self, x: Tensor) -> Tensor:
        if self.transpose_out:
            x = x.transpose(1, -1)  # batch, class, node else # batch, node, class
        return self.out_act(x)


class InteractionNet(AbsGraphFeatExtractor):
    r"""
    Implementation of the Interaction Graph-Network (https://arxiv.org/abs/1612.00222).
    Shown to be applicable for embedding many 4-momenta in e.g. https://arxiv.org/abs/1908.05318

    Receives column-wise data and returns column-wise

    Arguments:
        n_v: Number of vertices to expect per datapoint
        n_fpv: number features per vertex
        intfunc_outs: list of widths for the internal NN layers
        outfunc_outs: list of widths for the output NN layers
        do: dropout rate to be applied to hidden layers in the interaction-representation and post-interaction networks
        bn: whether batch normalisation should be applied to hidden layers in the interaction-representation and post-interaction networks
        act: activation function to apply to hidden layers in the interaction-representation and post-interaction networks
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`

    Examples::
        >>> inet = InteractionNet(n_v=128, n_fpv=10, intfunc_outs=[20,10], outfunc_outs=[20,4])
    """

    row_wise: Optional[bool] = False

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        intfunc_outs: List[int],
        outfunc_outs: List[int],
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
    ):
        super().__init__(
            n_v=n_v,
            n_fpv=n_fpv,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )
        store_attr(names="intfunc_outs,outfunc_outs")
        if not is_listy(self.intfunc_outs):
            self.intfunc_outs = [self.intfunc_outs]
        if not is_listy(self.outfunc_outs):
            self.outfunc_outs = [self.outfunc_outs]

        self.n_e = self.n_v * (self.n_v - 1)
        self.mat_rr, self.mat_rs = self._get_mat_r()
        self.mat_rr_t = self.mat_rr.t()
        self.fr = self._get_nn(2 * self.n_fpv, self.intfunc_outs)
        self.fo = self._get_nn(self.n_fpv + self.intfunc_outs[-1], self.outfunc_outs)

    def _get_mat_r(self) -> Tuple[Tensor, Tensor]:
        mat_rr, mat_rs = torch.zeros((self.n_v, self.n_e)), torch.zeros((self.n_v, self.n_e))
        for i in range(self.n_e):
            j = i % self.n_v
            mat_rr[j, (i + 1) % self.n_e] = 1
            mat_rs[j, i] = 1
        return to_device(mat_rr), to_device(mat_rs)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Learn new features per vertex

        Arguments:
            x: columnwise matrix data (batch x features x vertices)

        Returns:
            columnwise matrix data (batch x new features x vertices)
        """

        mat_o = torch.cat((x @ self.mat_rr, x @ self.mat_rs), 1)
        mat_o = self.fr(mat_o.transpose(1, 2)).transpose(1, 2)

        mat_o = mat_o @ self.mat_rr_t
        mat_o = torch.cat((x, mat_o), 1)

        mat_o = self.fo(mat_o.transpose(1, 2)).transpose(1, 2)
        return mat_o

    def get_out_size(self) -> Tuple[int, int]:
        return self.n_v, self.outfunc_outs[-1]


class GravNetLayer(AbsGraphBlock):
    r"""
    Single GravNet GNN layer (Qasim, Kieseler, Iiyama, & Pierini, 2019 https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9).
    Designed to be used as a sub-layer of a head block, e.g. :class:`~lumin.nn.models.blocks.head.GravNetHead`
    Passes features per vertex through NN to compute new features & coordinates of vertex in latent space.
    Vertex then receives additional features based on aggregations of distance-weighted features for k-nearest vertices in latent space
    Second NN transforms features per vertex.
    Input (batch x vertices x features) --> Output (batch x vertices x new features)

    Arguments:
        n_fpv: number of features per vertex to expect
        n_s: number of latent-spatial dimensions to compute
        n_lr: number of features to compute per vertex for latent representation
        k: number of neighbours (including self) each vertex should consider when aggregating latent-representation features
        agg_methods: list of functions to use to aggregate distance-weighted latent-representation features
        n_out: number of output features to compute per vertex
        cat_means: if True, will extend the incoming features per vertex by including the means of all features across all vertices
            :class:`~lumin.nn.models.block.head.GNNHead` aslo has a `cat_means` argument,
            which should be set to `False` if enabled here (otherwise averaging happens twice).
        f_slr_depth: number of layers to use for the latent rep. NN
        f_out_depth: number of layers to use for the output NN
        potential: function to control distance weighting (default is the exp(-d^2) potential used in the paper)
        use_sa: if true, will apply self-attention layer to the neighbourhhood features per vertex prior to aggregation
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is :class:`~lumin.nn.models.layers.batchnorms.LCBatchNorm1d`
        sa_class: class to use for self-attention layers, default is :class:`~lumin.nn.models.layers.self_attention.SelfAttention`
    """

    def __init__(
        self,
        n_fpv: int,
        n_s: int,
        n_lr: int,
        k: int,
        agg_methods: List[Callable[[Tensor], Tensor]],
        n_out: int,
        cat_means: bool = True,
        f_slr_depth: int = 1,
        f_out_depth: int = 1,
        potential: Optional[Callable[[Tensor], Tensor]] = None,
        use_sa: bool = False,
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
        sa_class: Callable[[int], nn.Module] = SelfAttention,
    ):
        super().__init__(
            n_v=None,
            n_fpv=n_fpv,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )
        store_attr(but=["potential"])
        if potential is not None:
            self.potential = potential
        if self.cat_means:
            self.n_fpv *= 2
        self.f_slr = self._get_nn(
            fan_in=self.n_fpv, width=2 * (self.n_s + self.n_lr), fan_out=self.n_s + self.n_lr, depth=self.f_slr_depth
        )
        self.f_out = self._get_nn(
            fan_in=self.n_fpv + (len(self.agg_methods) * self.n_lr),
            width=2 * self.n_out,
            fan_out=self.n_out,
            depth=self.f_out_depth,
        )
        if self.use_sa:
            self.sa_agg = self.sa_class(
                self.n_lr,
                self.n_lr // 4,
                do=self.do,
                bn=False,
                act=self.act,
                lookup_act=self.lookup_act,
                lookup_init=self.lookup_init,
            )

    @staticmethod
    def potential(x: Tensor) -> Tensor:
        return torch.exp(-10 * (x**2))

    def _get_nn(self, fan_in: int, width: int, fan_out: int, depth: int) -> nn.Module:
        return nn.Sequential(
            *[self._get_layer(fan_in if i == 0 else width, width if i + 1 < depth else fan_out) for i in range(depth)]
        )

    def _knn(self, dists: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        idxs = dists.argsort()
        i = np.arange(dists.shape[0])[:, None, None]
        j = np.arange(dists.shape[1])[None, :, None]
        return i, j, idxs[:, :, : self.k]

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Pass batch of vertices through GravNet layer and return new features per vertex

        Arguments:
            x: Incoming data (batch x vertices x features)

        Returns:
            Data with new features per vertex (batch x vertices x new features)
        """

        # Concat means
        if self.cat_means:
            x = torch.cat(
                [x, x.mean(1).unsqueeze(2).repeat_interleave(repeats=x.shape[1], dim=-1).transpose(1, 2)], dim=2
            )

        # Compute spatial and vertex features
        slr = self.f_slr(x)
        s, lr = slr[:, :, : self.n_s], slr[:, :, self.n_s :]

        # kNN
        d_jk = torch.norm(s[:, :, None] - s[:, None], dim=-1)
        idxs = self._knn(d_jk)
        d_jk = d_jk[idxs]
        lr = lr[:, None].expand(lr.shape[0], lr.shape[1], lr.shape[1], lr.shape[2])[idxs]

        # Aggregate feats
        v_jk = self.potential(d_jk)
        ft_ijk = lr * v_jk.unsqueeze(-1)
        if self.use_sa:
            ft_ijk = self.sa_agg(ft_ijk)
        fp = [x]
        for agg in self.agg_methods:
            fp.append(agg(ft_ijk))
        fp = torch.cat(fp, dim=-1)

        # Output
        return self.f_out(fp)

    def get_out_size(self) -> int:
        return self.n_out


class GravNet(AbsGraphFeatExtractor):
    r"""
    GravNet GNN head (Qasim, Kieseler, Iiyama, & Pierini, 2019 https://link.springer.com/article/10.1140/epjc/s10052-019-7113-9).
    Passes features per vertex (batch x vertices x features) through several :class:`~lumin.nn.models.blocks.head.GravNetLayer` layers.
    Like the paper model, this has the option of caching and concatenating the outputs of each GravNet layer prior to the final layer.
    The features per vertex are then flattened/aggregated across the vertices to flat data (batch x features).

    Arguments:
        n_v: Number of vertices to expect per datapoint
        n_fpv: number features per vertex
        cat_means: if True, will extend the incoming features per vertex by including the means of all features across all vertices
        f_slr_depth: number of layers to use for the latent rep. NN
        n_s: number of latent-spatial dimensions to compute
        n_lr: number of features to compute per vertex for latent representation
        k: number of neighbours (including self) each vertex should consider when aggregating latent-representation features
        f_out_depth: number of layers to use for the output NN
        n_out: number of output features to compute per vertex, if a list will add multiple gravnet layers, each of which outputs the respective number of features
        agg_methods: list of text representations of aggregation methods. Default is mean and max.
        gn_class: class to use for GravNet layers, default is :class:`~lumin.nn.models.blocks.gnn_blocks.GravNetLayer`
        use_sa: if true, will apply self-attention layer to the neighbourhhood features per vertex prior to aggregation
        do: dropout rate to be applied to hidden layers in the NNs
        bn: whether batch normalisation should be applied to hidden layers in the NNs
        act: activation function to apply to hidden layers in the NNs
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        freeze: whether to start with module parameters set to untrainable
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`
        sa_class: class to use for self-attention layers, default is :class:`~lumin.nn.models.layers.self_attention.SelfAttention`
    """

    row_wise: Optional[bool] = True

    def __init__(
        self,
        n_v: int,
        n_fpv: int,
        cat_means: bool,
        f_slr_depth: int,
        n_s: int,
        n_lr: int,
        k: int,
        f_out_depth: int,
        n_out: Union[List[int], int],
        agg_methods: Union[List[str], str] = ["mean", "max"],
        gn_class: Callable[[Dict[str, Any]], GravNetLayer] = GravNetLayer,
        use_sa: bool = False,
        do: float = 0,
        bn: bool = False,
        act: str = "relu",
        lookup_init: Callable[[str, Optional[int], Optional[int]], Callable[[Tensor], None]] = lookup_normal_init,
        lookup_act: Callable[[str], Any] = lookup_act,
        bn_class: Callable[[int], nn.Module] = nn.BatchNorm1d,
        sa_class: Callable[[int], nn.Module] = SelfAttention,
        **kargs,
    ):
        super().__init__(
            n_v=n_v,
            n_fpv=n_fpv,
            do=do,
            bn=bn,
            act=act,
            lookup_init=lookup_init,
            lookup_act=lookup_act,
            bn_class=bn_class,
        )
        store_attr(but=["n_v", "n_fpv", "do", "bn", "act", "lookup_init", "lookup_act", "bn_class"])
        if not is_listy(self.n_out):
            self.n_out = [self.n_out]
        self._check_agg_methods(agg_methods)
        self.grav_layers = self._get_grav_layers()
        self.out_sz = (self.n_v, np.sum([l.get_out_size() for l in self.grav_layers]))

    @staticmethod
    def _agg_mean(x: Tensor) -> Tensor:
        return torch.mean(x, dim=2)

    @staticmethod
    def _agg_max(x: Tensor) -> Tensor:
        return torch.max(x, dim=2)[0]

    @staticmethod
    def _agg_absmax(x: Tensor) -> Tensor:
        return torch.max(x.abs(), dim=2)[0]

    def _check_agg_methods(self, agg_methods: Union[List[str], str]) -> None:
        self.agg_methods = []
        if not is_listy(agg_methods):
            agg_methods = [agg_methods]
        for m in agg_methods:
            m = m.lower()
            if m == "mean":
                self.agg_methods.append(self._agg_mean)
            elif m == "max":
                self.agg_methods.append(self._agg_max)
            elif m == "absmax":
                self.agg_methods.append(self._agg_absmax)
            else:
                raise ValueError(f"{m} not in [mean, max, absmax]")

    def _get_grav_layers(self) -> nn.ModuleList:
        ls = []
        gl = partial(
            self.gn_class,
            f_slr_depth=self.f_slr_depth,
            n_s=self.n_s,
            n_lr=self.n_lr,
            k=self.k,
            agg_methods=self.agg_methods,
            f_out_depth=self.f_out_depth,
            cat_means=self.cat_means,
            use_sa=self.use_sa,
            do=self.do,
            bn=self.bn,
            act=self.act,
            lookup_init=self.lookup_init,
            lookup_act=self.lookup_act,
            bn_class=self.bn_class,
            sa_class=self.sa_class,
        )
        n_fpv = self.n_fpv
        for n in self.n_out:
            ls.append(gl(n_fpv=n_fpv, n_out=n))
            n_fpv = ls[-1].get_out_size()
        return nn.ModuleList(ls)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Passes input through the GravNet head.

        Arguments:
            x: row-wise tensor (batch x vertices x features)

        Returns:
            Resulting tensor row-wise tensor (batch x vertices x new features)
        """

        outs = [x]
        for l in self.grav_layers:
            outs.append(l(outs[-1]))
        return torch.cat(outs[1:], dim=-1)

    def get_out_size(self) -> Tuple[int, int]:
        return self.out_sz
