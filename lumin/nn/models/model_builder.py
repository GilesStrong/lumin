
from typing import Dict, Union, Any, Callable, Tuple, Optional, List, Iterator
import pickle
import math
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
import torch

from .layers.activations import lookup_act
from .initialisations import lookup_normal_init
from .helpers import CatEmbedder
from .blocks.body import FullyConnected, AbsBody
from .blocks.head import CatEmbHead, AbsHead
from .blocks.tail import ClassRegMulti, AbsTail
from ..losses.basic_weighted import WeightedCCE, WeightedMSE
from ...utils.misc import to_device

__all__ = ['ModelBuilder']

'''
Todo
- Better typing for nn._WeightedLoss
'''


class ModelBuilder(object):
    r'''
    Class to build models to specified architecture on demand along with an optimiser.

    Arguments:
        objective: string representation of network objective, i.e. 'classification', 'regression', 'multiclass'
        n_out: number of outputs required
        cont_feats: list of names of continuous input features
        model_args: dictionary of dictionaries of keyword arguments to pass to head, body, and tail to control architrcture
        opt_args: dictionary of arguments to pass to optimiser. Missing kargs will be filled with default values.
            Currently, only ADAM (default), RAdam, Ranger, and SGD are available.
        cat_embedder: :class:`~lumin.nn.models.helpers.CatEmbedder` for embedding categorical inputs
        cont_subsample_rate: if between in range (0, 1), will randomly select a fraction of continuous features (rounded upwards) to use as inputs
        guaranteed_feats: if subsampling features, will always include the features listed here, which count towards the subsample fraction
        loss: either and uninstantiated loss class, or leave as 'auto' to select loss according to objective
        head: uninstantiated class which can receive input data and upscale it to model width
        body: uninstantiated class which implements the main bulk of the model's hidden layers
        tail: uninstantiated class which scales the body to the required number of outputs and implements any final activation function and output scaling
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        pretrain_file: if set, will load saved parameters for entire network from saved model
        freeze_head: whether to start with the head parameters set to untrainable
        freeze_body: whether to start with the body parameters set to untrainable


    Examples::
        >>> model_builder = ModelBuilder(objective='classifier',
        >>>                              cont_feats=cont_feats, n_out=1,
        >>>                              model_args={'body':{'depth':4,
        >>>                                                  'width':100}})
        >>>
        >>> min_targs = np.min(targets, axis=0).reshape(targets.shape[1],1)
        >>> max_targs = np.max(targets, axis=0).reshape(targets.shape[1],1)
        >>> min_targs[min_targs > 0] *=0.8
        >>> min_targs[min_targs < 0] *=1.2
        >>> max_targs[max_targs > 0] *=1.2
        >>> max_targs[max_targs < 0] *=0.8
        >>> y_range = np.hstack((min_targs, max_targs))
        >>> model_builder = ModelBuilder(
        >>>     objective='regression', cont_feats=cont_feats, n_out=6,
        >>>     cat_embedder=CatEmbedder.from_fy(train_fy),
        >>>     model_args={'body':{'depth':4, 'width':100},
        >>>                 'tail':{y_range=y_range})
        >>>
        >>> model_builder = ModelBuilder(objective='multiclassifier',
        >>>                              cont_feats=cont_feats, n_out=5,
        >>>                              model_args={'body':{'width':100,
        >>>                                                  'depth':6,
        >>>                                                  'do':0.1,
        >>>                                                  'res':True}})
        >>>
        >>> model_builder = ModelBuilder(objective='classifier',
        >>>                              cont_feats=cont_feats, n_out=1,
        >>>                              model_args={'body':{'depth':4,
        >>>                                                  'width':100}},
        >>>                              opt_args={'opt':'sgd',
        >>>                                        'momentum':0.8,
        >>>                                        'weight_decay':1e-5},
        >>>                              loss=partial(SignificanceLoss,
        >>>                                           sig_weight=sig_weight,
        >>>                                           bkg_weight=bkg_weight,
        >>>                                           func=calc_ams_torch))
    '''

    # TODO: Make opt use partials rather than strings
    # TODO: Classmethod from_fy
    # TODO: Check examples
    # TODO: Factor out cat_embedder to head
    # TODO: Add matrix example
    # TODO: Add partial block example

    def __init__(self, objective:str, n_out:int, cont_feats:Optional[List[str]]=None,
                 model_args:Optional[Dict[str,Dict[str,Any]]]=None, opt_args:Optional[Dict[str,Any]]=None, cat_embedder:Optional[CatEmbedder]=None,
                 cont_subsample_rate:Optional[float]=None, guaranteed_feats:Optional[List[str]]=None, loss:Any='auto',
                 head:Callable[[Any],AbsHead]=CatEmbHead, body:Callable[[Any],AbsBody]=FullyConnected, tail:Callable[[Any],AbsTail]=ClassRegMulti,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],nn.Module]=lookup_act, pretrain_file:Optional[str]=None,
                 freeze_head:bool=False, freeze_body:bool=False, freeze_tail:bool=False):
        self.objective,self.cont_feats,self.n_out,self.cat_embedder = objective.lower(),cont_feats,n_out,cat_embedder
        self.cont_subsample_rate,self.guaranteed_feats = cont_subsample_rate,guaranteed_feats
        self.head,self.body,self.tail = head,body,tail
        self.lookup_init,self.lookup_act,self.pretrain_file, = lookup_init,lookup_act,pretrain_file
        self.freeze_head,self.freeze_body,self.freeze_tail = freeze_head,freeze_body,freeze_tail
        self._parse_loss(loss)
        self._parse_model_args(model_args)
        self._parse_opt_args(opt_args)
        self._subsample()

    @classmethod
    def from_model_builder(cls, model_builder, pretrain_file:Optional[str]=None, freeze_head:bool=False, freeze_body:bool=False, freeze_tail:bool=False,
                           loss:Optional[Any]=None, opt_args:Optional[Dict[str,Any]]=None):
        r'''
        Instantiate a :class:`~lumin.nn.models.model_builder.ModelBuilder` from an exisitng :class:`~lumin.nn.models.model_builder.ModelBuilder`, but with options to adjust loss, optimiser, pretraining, and module freezing

        Arguments:
            model_builder: existing :class:`~lumin.nn.models.model_builder.ModelBuilder` or filename for a pickled :class:`~lumin.nn.models.model_builder.ModelBuilder` 
            pretrain_file: if set, will load saved parameters for entire network from saved model
            freeze_head: whether to start with the head parameters set to untrainable
            freeze_body: whether to start with the body parameters set to untrainable
            freeze_tail: whether to start with the tail parameters set to untrainable
            loss: either and uninstantiated loss class, or leave as 'auto' to select loss according to objective            
            opt_args: dictionary of arguments to pass to optimiser. Missing kargs will be filled with default values. Choice of optimiser (`'opt'`) keyword can
                either be set by passing the string name (e.g. `'adam'` ), but only ADAM and SGD are available this way, or by passing an uninstantiated
                optimiser (e.g. torch.optim.Adam). If no optimser is set, then it defaults to ADAM. Additional keyword arguments can be set, and these will be
                passed tot he optimiser during instantiation

        Returns:
            Instantiated :class:`~lumin.nn.models.model_builder.ModelBuilder`
            
        Examples::
            >>> new_model_builder = ModelBuilder.from_model_builder(
            >>>     ModelBuidler)
            >>>
            >>> new_model_builder = ModelBuilder.from_model_builder(
            >>>     ModelBuidler, loss=partial(
            >>>         SignificanceLoss, sig_weight=sig_weight,
            >>>         bkg_weight=bkg_weight, func=calc_ams_torch))
            >>>
            >>> new_model_builder = ModelBuilder.from_model_builder(
            >>>     'weights/model_builder.pkl',
            >>>     opt_args={'opt':'sgd', 'momentum':0.8, 'weight_decay':1e-5})
            >>>
            >>> new_model_builder = ModelBuilder.from_model_builder(
            >>>     'weights/model_builder.pkl',
            >>>     opt_args={'opt':torch.optim.Adam,
            ...               'momentum':0.8,
            ...               'weight_decay':1e-5})
        '''

        if isinstance(model_builder, str):
            with open(model_builder, 'rb') as fin: model_builder = pickle.load(fin)
        model_args = {'head': model_builder.head_kargs, 'body': model_builder.body_kargs, 'tail': model_builder.tail_kargs}
        if not hasattr(model_builder, 'cont_subsample_rate'): model_builder.cont_subsample_rate,model_builder.guaranteed_feats = None,None  # <0.3.2 compat
        return cls(objective=model_builder.objective, cont_feats=model_builder.cont_feats, n_out=model_builder.n_out,
                   cat_embedder=model_builder.cat_embedder, model_args=model_args, opt_args=opt_args if opt_args is not None else {},
                   cont_subsample_rate=model_builder.cont_subsample_rate, guaranteed_feats=model_builder.guaranteed_feats,
                   loss=model_builder.loss if loss is None else loss, head=model_builder.head, body=model_builder.body, tail=model_builder.tail,
                   pretrain_file=pretrain_file, freeze_head=freeze_head, freeze_body=freeze_body, freeze_tail=freeze_tail)
            
    def _parse_loss(self, loss:Union[Any,'auto']='auto') -> None:
        if loss == 'auto':
            if 'class' in self.objective:
                if self.n_out > 1 and 'multiclass' in self.objective: self.loss = WeightedCCE
                else:                                                 self.loss = nn.BCELoss
            else:
                self.loss = WeightedMSE
        else:   
            self.loss = loss

    def _parse_model_args(self, model_args:Optional[Dict[str,Any]]=None) -> None:
        if model_args is None: model_args = {}
        else:                  model_args = {k.lower(): model_args[k] for k in model_args}
        self.head_kargs = {} if model_args is None or 'head' not in model_args else model_args['head']
        self.body_kargs = {} if model_args is None or 'body' not in model_args else model_args['body']
        self.tail_kargs = {} if model_args is None or 'tail' not in model_args else model_args['tail']
    
    def _parse_opt_args(self, opt_args:Optional[Dict[str,Any]]=None) -> None:
        if opt_args is None: opt_args = {}
        else:                opt_args = {k.lower(): opt_args[k] for k in opt_args}
        self.opt_args = {k: opt_args[k] for k in opt_args if k != 'opt'}
        if 'opt' not in opt_args:
            print('No optimiser specified, defaulting to ADAM')
            self.opt = optim.Adam
        else:
            self.opt = opt_args['opt'] if not isinstance(opt_args['opt'], str) else self._interp_opt(opt_args['opt'])

    def _subsample(self) -> None:
        if self.cont_subsample_rate is not None:
            self.use_conts = self.guaranteed_feats if self.guaranteed_feats is not None else []
            n = math.ceil(len(self.cont_feats)*self.cont_subsample_rate)-len(self.use_conts)
            np.random.seed()  # Is this necessary?
            self.use_conts += list(np.random.choice([f for f in self.cont_feats if f not in self.use_conts], size=n, replace=False))
            self.n_cont_in = len(self.use_conts)
            cont_idxs = np.array([self.cont_feats.index(f) for f in self.use_conts])
            cat_idxs  = len(self.cont_feats)+np.arange(self.cat_embedder.n_cat_in)
            self.input_mask = np.hstack((cont_idxs, cat_idxs))
            self.input_mask.sort()
            
        else:
            self.use_conts = self.cont_feats
            self.n_cont_in = len(self.cont_feats)
            self.input_mask = None

    @staticmethod
    def _interp_opt(opt:str) -> Callable[[Iterator, Optional[Any]], optim.Optimizer]:
        opt = opt.lower()
        if   opt == 'adam':   return optim.Adam
        elif opt == 'sgd':    return optim.SGD
        else: raise ValueError(f"Optimiser {opt} not interpretable from string, please pass as class")

    def _build_opt(self, model:nn.Module) -> optim.Optimizer:
        if isinstance(self.opt, str): self.opt = self._interp_opt(self.opt)  # Backwards compatability with pre-v0.3.1 saves
        return self.opt(model.parameters(), **self.opt_args)

    def set_lr(self, lr:float) -> None:
        r'''
        Set learning rate for all model parameters

        Arguments:
            lr: learning rate
        '''
        
        self.opt_args['lr'] = lr

    def get_head(self) -> AbsHead:
        r'''
        Construct head module

        Returns:
            Instantiated head nn.Module
        '''

        if not hasattr(self, 'use_conts'):
            self.use_conts = self.cont_feats  # Backwards compatability with pre-v0.3.2 saves
            self.input_mask = None

        return self.head(cont_feats=self.use_conts, cat_embedder=self.cat_embedder, lookup_init=self.lookup_init, freeze=self.freeze_head, **self.head_kargs)

    def get_body(self, n_in:int, feat_map:List[str]) -> AbsBody:
        r'''
        Construct body module

        Returns:
            Instantiated body nn.Module
        '''

        return self.body(n_in=n_in, feat_map=feat_map, lookup_init=self.lookup_init, lookup_act=self.lookup_act, freeze=self.freeze_body, **self.body_kargs)

    def get_tail(self, n_in:int) -> nn.Module:
        r'''
        Construct tail module

        Returns:
            Instantiated tail nn.Module
        '''

        return self.tail(n_in=n_in, n_out=self.n_out, objective=self.objective, lookup_init=self.lookup_init, freeze=self.freeze_tail, **self.tail_kargs)

    def build_model(self) -> nn.Module:
        r'''
        Construct entire network module

        Returns:
            Instantiated nn.Module
        '''

        head = self.get_head()
        body = self.get_body(head.get_out_size(), head.feat_map)
        tail = self.get_tail(body.get_out_size())
        return nn.Sequential(head, body, tail)

    def load_pretrained(self, model:nn.Module):
        r'''
        Load model weights from pretrained file

        Arguments:
            model: instantiated model, i.e. return of :meth:`~lumin.nn.models.model_builder.ModelBuilder.build_model`

        Returns:
            model with weights loaded
        '''

        state = torch.load(self.pretrain_file, map_location='cpu')
        print('Loading pretrained model')
        return model.load_state_dict(state['model'])

    def get_model(self) -> Tuple[nn.Module, optim.Optimizer, Any]:
        r'''
        Construct model, loss, and optimiser, optionally loading pretrained weights

        Returns:
            Instantiated network, optimiser linked to model parameters, and uninstantiated loss
        '''

        model = self.build_model()
        if self.pretrain_file is not None: self.load_pretrained(model)
        model = to_device(model)
        opt = self._build_opt(model)
        return model, opt, self.loss, self.input_mask

    def get_out_size(self) -> int:
        r'''
        Get number of outputs of model

        Returns:
            number of outputs of network
        '''
        
        return self.tail.get_out_size()
