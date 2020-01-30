r'''
This file contains code modfied from https://github.com/ducha-aiki/LSUV-pytorch which is made available under the following BSD 2-Clause "Simplified" Licence:
Copyright (C) 2017, Dmytro Mishkin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Apache Licence 2.0 underwhich the majority of the rest of LUMIN is distributed does not apply to the code within this file.
'''

import numpy as np
from typing import Optional, Union, Tuple

import torch
import torch.nn.init
import torch.nn as nn
from torch import Tensor

from lumin.nn.callbacks.callback import Callback
from lumin.nn.data.batch_yielder import BatchYielder
from lumin.nn.models.abs_model import AbsModel

__all__ = ['LsuvInit']


class LsuvInit(Callback):
    r'''
    Applies Layer-Sequential Unit-Variance (LSUV) initialisation to model, as per Mishkin & Matas 2016 https://arxiv.org/abs/1511.06422.
    When training begins for the first time, `Conv1D`, `Conv2D`, and `Linear` modules in the model will be LSUV initialised using the BatchYielder inputs.
    This involves initialising the weights with orthonormal matirces and then iteratively scaling them such that the stadndar deviation of the layer outputs is
    equal to a desired value, within some tolerance.

    Arguments:
        needed_std: desired standard deviation of layer outputs
        std_tol: tolerance for matching standard deviation with target
        max_attempts: number of times to attempt weight scaling per layer
        do_orthonorm: whether to apply orthonormal initialisation first, or rescale the exisiting values
        verbose: whether to print out details of the rescaling
        model: :class:`~lumin.nn.models.model.Model` to provide parameters, alternatively call :meth:`~lumin.nn.models.Model.set_model`

    Example::
        >>> lsuv = LsuvInit()
        >>>
        >>> lsuv = LsuvInit(verbose=True)
        >>> 
        >>> lsuv = LsuvInit(needed_std=0.5, std_tol=0.01, max_attempts=100, do_orthonorm=True)
    '''

    def __init__(self, needed_std:float=1.0, std_tol:float=0.1, max_attempts:int=10, do_orthonorm:bool=True, verbose:bool=False, model:Optional[AbsModel]=None):
        super().__init__(model=model)
        self.needed_std,self.std_tol,self.max_attempts,self.do_orthonorm,self.verbose = needed_std,std_tol,max_attempts,do_orthonorm,verbose
    
    def on_train_begin(self, **kargs) -> None:
        r'''
        Sets the callback to initialise the model the first time that `on_epoch_begin` is called.
        '''

        self.init = False
        self.gg = {'hook_position':0, 'total_fc_conv_layers':0,'done_counter':-1,'hook':None,'act_dict':{},'counter_to_apply_correction':0,
                   'correction_needed':False,'current_coef':1.0}
        
    def on_epoch_begin(self, by:BatchYielder, **kargs) -> None:
        r'''
        If the LSUV process has yet to run, then it will run using all of the input data provided by the `BatchYielder`

        Arguments:
            by: BatchYielder providing data for the upcoming epoch
        '''

        if not self.init:
            print('Running LSUV initialisation')
            self._run_lsuv(by.get_inputs(on_device=True))
            self.init = True
    
    @staticmethod
    def _svd_orthonormal(w:np.ndarray) -> np.ndarray:
        shape = w.shape
        if len(shape) < 2: raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q.astype(np.float32)

    def _store_activations(self, module:nn.Module, input:Tensor, output:Tensor):
        self.gg['act_dict'] = output.data.cpu().numpy()

    def _add_current_hook(self, m:nn.Module) -> None:
        if self.gg['hook'] is not None: return
        if self._check_layer(m):
            if self.gg['hook_position'] > self.gg['done_counter']: self.gg['hook'] = m.register_forward_hook(self._store_activations)
            else:                                                  self.gg['hook_position'] += 1

    def _count_conv_fc_layers(self, m:nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear): self.gg['total_fc_conv_layers'] += 1
            
    @staticmethod
    def _check_layer(m:nn.Module) -> bool:
        return isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear)
    
    def _orthogonal_weights_init(self, m:nn.Module) -> None:
        if self._check_layer(m):
            if hasattr(m, 'weight_v'):
                w_ortho = self._svd_orthonormal(m.weight_v.data.cpu().numpy())
                m.weight_v.data = torch.from_numpy(w_ortho)
            else:
                w_ortho = self._svd_orthonormal(m.weight.data.cpu().numpy())
                m.weight.data = torch.from_numpy(w_ortho)
            if hasattr(m, 'bias'): nn.init.zeros_(m.bias)

    def _apply_weights_correction(self, m:nn.Module) -> None:
        if self.gg['hook'] is None: return
        if not self.gg['correction_needed']: return
        if self._check_layer(m):
            if self.gg['counter_to_apply_correction'] < self.gg['hook_position']:
                self.gg['counter_to_apply_correction'] += 1
            else:
                if hasattr(m, 'weight_g'):
                    m.weight_g.data *= float(self.gg['current_coef'])
                    self.gg['correction_needed'] = False
                else:
                    m.weight.data *= self.gg['current_coef']
                    self.gg['correction_needed'] = False

    def _run_lsuv(self, data:Union[Tensor,Tuple[Tensor,Tensor]]) -> None:
        # cuda = next(self.model.model.parameters()).is_cuda
        self.model.model.eval()
        self.model.model.apply(self._count_conv_fc_layers)
        if self.verbose: print(f'Total layers to process: {self.gg["total_fc_conv_layers"]}')
        if self.do_orthonorm:
            self.model.model.apply(self._orthogonal_weights_init)
            if self.verbose: print('Orthonorm done')
            # if cuda: self.model.model = self.model.model.cuda()
        for layer_idx in range(self.gg['total_fc_conv_layers']):
            if self.verbose: print(f'Checking layer {layer_idx}')
            self.model.model.apply(self._add_current_hook)
            self.model.model(data)
            current_std = self.gg['act_dict'].std()
            if self.verbose: print(f'std at layer {layer_idx} = {current_std}')
            attempts = 0
            while np.abs(current_std-self.needed_std) > self.std_tol:
                self.gg['current_coef'] = self.needed_std/(current_std+1e-8)
                self.gg['correction_needed'] = True
                self.model.model.apply(self._apply_weights_correction)
                self.model.model(data)
                current_std = self.gg['act_dict'].std()
                if self.verbose: print(f'std at layer {layer_idx} = {current_std} mean = {self.gg["act_dict"].mean()}')
                attempts += 1
                if attempts > self.max_attempts:
                    print(f'Cannot converge in {self.max_attempts} iterations')
                    break
            if self.gg['hook'] is not None: self.gg['hook'].remove()
            self.gg['done_counter'] += 1
            self.gg['counter_to_apply_correction'] = 0
            self.gg['hook_position'] = 0
            self.gg['hook'] = None
            if self.verbose: print(f'Initialised layer {layer_idx}')
        if self.verbose: print('LSUV init done!')
        # if not cuda: self.model.model = self.model.model.cpu()
