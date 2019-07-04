
import numpy as np
from typing import Dict, Union, Any, Callable, Tuple, Optional
import pickle
import  warnings

import torch.nn as nn
import torch.optim as optim
from torch.tensor import Tensor
import torch

from .layers.activations import lookup_act
from .initialisations import lookup_normal_init
from .helpers import Embedder
from .blocks.body import FullyConnected
from .blocks.head import CatEmbHead
from .blocks.tail import ClassRegMulti
from ..losses.basic_weighted import WeightedCCE, WeightedMSE

'''
Todo
- Better typing for nn._WeightedLoss
'''


class ModelBuilder(object):
    r'''
    Class to build models to specified architecture on demand along with an optimiser.
    For regression tasks, y_range can be set with per-output minima and maxima. The outputs are then adjusted according to ((y_max-y_min)*x)+self.y_min, where x
    is the output of the network passed through a sigmoid function. Effectively allowing regression to be performed without normalising and standardising the
    target values. Note it is safest to allow some leaway in setting the min and max, e.g. max = 1.2*max, min = 0.8*min 
    Output activation function is automatically set according to objective and y_range.

    Arguments:
        objective: string representation of network objective, i.e. 'classification', 'regression', 'multiclass'
        n_cont_in: number of continuous inputs to expect
        n_out: number of outputs required
        y_range: if not None, will apply rescaling to network outputs.
        model_args: dictionary of arguments to pass to head, body, and tail to control architrcture. Missing kargs will be filled with default values.
            Full list is:
                'width': numer of neurons in hidden layers
                'depth': number of hidden layers in entire model
                'do': Dropout rate for main Dropout layers
                'do_cat': Dropout rate for embedded categorical features
                'do_cont': Dropout rate for continuous features
                'bn': whether to use batch normalisation
                'act': string representation of internal activation functions
                'res': whether to use residual skip connections
                'dense': whether to use layer-wise concatination
                'growth_rate': rate at which width of dense layers should increase with depth beyond the initial layer. Ignored if res=True. Can be negative.
        opt_args: dictionary of arguments to pass to optimiser. Missing kargs will be filled with default values.
            Currently, only ADAM (default) and SGD are available.
        cat_embedder: :class:Embedder for embedding categorical inputs
        loss: either and uninstantiated loss class, or leave as 'auto' to select loss according to objective
        head: uninstantiated class which can receive input data and upscale it to model width
        body: uninstantiated class which implements the main bulk of the model's hidden layers
        tail: uninstantiated class which scales the body to the required number of outputs and implements any final activation function and output scaling
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        pretrain_file: if set, will load saved parameters for entire network from saved model
        freeze_head: whether to start with the head parameters set to untrainable
        freeze_body: whether to start with the body parameters set to untrainable
        cat_args: depreciated in place of cat_embedder

    Examples::
        >>> model_builder = ModelBuilder(objective='classifier', n_cont_in=30, n_out=1)
        >>> min_targs = np.min(targets, axis=0).reshape(targets.shape[1],1)
            max_targs = np.max(targets, axis=0).reshape(targets.shape[1],1)
            min_targs[min_targs > 0] *=0.8
            min_targs[min_targs < 0] *=1.2
            max_targs[max_targs > 0] *=1.2
            max_targs[max_targs < 0] *=0.8
            y_range = np.hstack((min_targs, max_targs))
            model_builder = ModelBuilder(objective='regression', n_cont_in=30, n_out=6, cat_embedder=Embedder.from_fy(train_fy), y_range=y_range)
        >>> model_builder = ModelBuilder(objective='multiclassifier', n_cont_in=30, n_out=5, model_args={'depth':6, do':0.1, 'res':True})
        >>> model_builder = ModelBuilder(objective='classifier', n_cont_in=30, n_out=1, opt_args={'opt':'sgd', 'momentum':0.8, 'weight_decay':1e-5},
                                         loss=partial(SignificanceLoss, sig_weight=sig_weight, bkg_weight=bkg_weight, func=calc_ams_torch))
    '''

    # TODO: Make opt use partials rather than strings
    # TODO: Classmethod from_fy

    def __init__(self, objective:str, n_cont_in:int, n_out:int, y_range:Optional[Union[Tuple,np.ndarray]]=None,
                 model_args:Optional[Dict[str,Any]]=None, opt_args:Optional[Dict[str,Any]]=None, cat_embedder:Optional[Embedder]=None,
                 loss:Union[Any,'auto']='auto', head:nn.Module=CatEmbHead, body:nn.Module=FullyConnected, tail:nn.Module=ClassRegMulti,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],nn.Module]=lookup_act, pretrain_file:Optional[str]=None, freeze_head:bool=False, freeze_body:bool=False,
                 cat_args:Dict[str,Any]=None):
        self.objective,self.n_cont_in,self.n_out,self.y_range,self.cat_embedder = objective.lower(),n_cont_in,n_out,y_range,cat_embedder
        self.head,self.body,self.tail = head,body,tail
        self.lookup_init,self.lookup_act,self.pretrain_file,self.freeze_head,self.freeze_body = lookup_init,lookup_act,pretrain_file,freeze_head,freeze_body
        self._parse_loss(loss)
        self._parse_model_args(model_args)
        self._parse_opt_args(opt_args)
        # XXX Remove in v0.3
        if self.cat_embedder is None and  cat_args is not None:
            warnings.warn('''Passing cat_args (dictionary of lists for embedding categorical features) is depreciated and will be removed in v0.3.
                             Please move to passing an Embedder class to cat_embedder''')
            cat_args = {k:cat_args[k] for k in cat_args if k != 'n_cat_in'}
            if 'emb_szs' in cat_args: cat_args['emb_szs'] = cat_args['emb_szs'][:,-1]
            self.cat_embedder = Embedder(**cat_args)

    @classmethod
    def from_model_builder(cls, model_builder, pretrain_file:Optional[str]=None, freeze_head:bool=False, freeze_body:bool=False,
                           loss:Optional[Any]=None, opt_args:Optional[Dict[str,Any]]=None):
        r'''
        Instantiate a :class:ModelBuilder from an exisitng :class:ModelBuilder, but with options to adjust loss, optimiser, pretraining, and module freezing

        Arguments:
            model_builder: existing :class:ModelBuilder or filename for a pickled :class:ModelBuilder 
            pretrain_file: if set, will load saved parameters for entire network from saved model
            freeze_head: whether to start with the head parameters set to untrainable
            freeze_body: whether to start with the body parameters set to untrainable
            loss: either and uninstantiated loss class, or leave as 'auto' to select loss according to objective            
            opt_args: dictionary of arguments to pass to optimiser. Missing kargs will be filled with default values.
                Currently, only ADAM (default) and SGD are available.

        Returns:
            Instantiated :class:ModelBuilder
            
        Examples::
            >>> new_model_builder = ModelBuilder.from_model_builder(ModelBuidler)
            >>> new_model_builder = ModelBuilder.from_model_builder(ModelBuidler, loss=partial(SignificanceLoss, sig_weight=sig_weight,
                                                                                               bkg_weight=bkg_weight, func=calc_ams_torch))
            >>> new_model_builder = ModelBuilder.from_model_builder('weights/model_builder.pkl', opt_args={'opt':'sgd', 'momentum':0.8, 'weight_decay':1e-5})
        '''

        if isinstance(model_builder, str):
            with open(model_builder, 'rb') as fin: model_builder = pickle.load(fin)
        model_args = {'width': model_builder.width, 'depth': model_builder.depth, 'do': model_builder.do, 'do_cat': model_builder.do_cat,
                      'do_cont': model_builder.do_cont, 'bn': model_builder.bn, 'act': model_builder.act, 'res': model_builder.res,
                      'dense': model_builder.dense, 'growth_rate': model_builder.growth_rate}
        return cls(objective=model_builder.objective, n_cont_in=model_builder.n_cont_in, n_out=model_builder.n_out, y_range=model_builder.y_range,
                   cat_embedder=model_builder.cat_embedder, model_args=model_args, opt_args=opt_args if opt_args is not None else {},
                   loss=model_builder.loss if loss is None else loss, head=model_builder.head, body=model_builder.body, tail=model_builder.tail,
                   pretrain_file=pretrain_file, freeze_head=freeze_head, freeze_body=freeze_body)
            
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
        model_args       = {k.lower(): model_args[k] for k in model_args}
        self.width       = 100    if model_args is None or 'width'       not in model_args else model_args['width']
        self.depth       = 4      if model_args is None or 'depth'       not in model_args else model_args['depth']
        self.do          = 0      if model_args is None or 'do'          not in model_args else model_args['do']
        self.do_cat      = 0      if model_args is None or 'do_cat'      not in model_args else model_args['do_cat']
        self.do_cont     = 0      if model_args is None or 'do_cont'     not in model_args else model_args['do_cont']
        self.bn          = False  if model_args is None or 'bn'          not in model_args else model_args['bn']
        self.act         = 'relu' if model_args is None or 'act'         not in model_args else model_args['act'].lower()
        self.res         = False  if model_args is None or 'res'         not in model_args else model_args['res']
        self.dense       = False  if model_args is None or 'dense'       not in model_args else model_args['dense']
        self.growth_rate = 0      if model_args is None or 'growth_rate' not in model_args else model_args['growth_rate']
    
    def _parse_opt_args(self, opt_args:Optional[Dict[str,Any]]=None) -> None:
        if opt_args is None: opt_args = {}
        else:                opt_args = {k.lower(): opt_args[k] for k in opt_args}
        self.opt_args = {k: opt_args[k] for k in opt_args if k != 'opt'}   
        self.opt = 'adam' if 'opt' not in opt_args else opt_args['opt']

    def _build_opt(self, model:nn.Module) -> optim.Optimizer:
        if   self.opt == 'adam':      return optim.Adam(model.parameters(), **self.opt_args)
        elif self.opt == 'sgd':       return optim.SGD(model.parameters(),  **self.opt_args)
        else: raise ValueError(f"Optimiser {self.opt} not currently available")

    def set_lr(self, lr:float) -> None:
        r'''
        Set learning rate for all model parameters
        '''
        
        self.opt_args['lr'] = lr

    def get_head(self) -> nn.Module:
        r'''
        Construct head module

        Returns:
            Instantiated head nn.Module
        '''

        return self.head(n_cont_in=self.n_cont_in, act=self.act, do=self.do, do_cont=self.do_cont, do_cat=self.do_cat, bn=self.bn,
                         cat_embedder=self.cat_embedder, lookup_init=self.lookup_init, lookup_act=self.lookup_act, freeze=self.freeze_head)

    def get_body(self, n_in:int) -> nn.Module:
        r'''
        Construct body module

        Returns:
            Instantiated body nn.Module
        '''

        return self.body(n_in=n_in, depth=self.depth, width=self.width, do=self.do, bn=self.bn, act=self.act, res=self.res, dense=self.dense,
                         growth_rate=self.growth_rate, lookup_init=self.lookup_init, lookup_act=self.lookup_act, freeze=self.freeze_body)

    def get_tail(self, n_in) -> nn.Module:
        r'''
        Construct tail module

        Returns:
            Instantiated tail nn.Module
        '''

        return self.tail(n_in=n_in, n_out=self.n_out, objective=self.objective, y_range=self.y_range,lookup_init=self.lookup_init)

    def build_model(self) -> nn.Module:
        r'''
        Construct entire network module

        Returns:
            Instantiated nn.Module
        '''

        head = self.get_head()
        if hasattr(head, 'get_out_size'):
            out_size = head.get_out_size()
        else:
            *_, last = head.parameters()
            out_size = len(last)
        body = self.get_body(out_size)
        if hasattr(body, 'get_out_size'):
            out_size = body.get_out_size()
        else:
            *_, last = body.parameters()
            out_size = len(last)
        tail = self.get_tail(out_size)
        return nn.Sequential(head, body, tail)

    def load_pretrained(self, model:nn.Module):
        r'''
        Load model weights from pretrained file

        Arguments:
            model: instantiated model, i.e. return of :meth:build_model

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
        opt = self._build_opt(model)
        return model, opt, self.loss

    def get_out_size(self) -> int:
        r'''
        Get number of outputs of model

        Returns:
            number of outputs of network
        '''
        
        return self.tail.get_out_size()
