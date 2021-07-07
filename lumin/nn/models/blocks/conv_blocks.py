from typing import Callable, Union, Optional, Any, Tuple
from fastcore.all import store_attr
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from ..initialisations import lookup_normal_init
from ..layers.activations import lookup_act


__all__ = ['Conv1DBlock', 'Res1DBlock', 'ResNeXt1DBlock', 'AdaptiveAvgMaxConcatPool1d', 'AdaptiveAvgMaxConcatPool2d', 'AdaptiveAvgMaxConcatPool3d', 'SEBlock1d',
           'SEBlock2d', 'SEBlock3d']


class Conv1DBlock(nn.Module):
    r'''
    Basic building block for a building and applying a single 1D convolutional layer.

    Arguments:
        in_c: number of input channels (number of features per object / rows in input matrix)
        out_c: number of output channels (number of features / rows in output matrix)
        kernel_sz: width of kernel, i.e. the number of columns to overlay
        padding: amount of padding columns to add at start and end of convolution.
            If left as 'auto', padding will be automatically computed to conserve the number of columns.
        stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
            stride 2 = kernel centred on ever other column and input size halved, et cetera.
        act: string representation of argument to pass to lookup_act
        bn: whether to use batch normalisation (default order weights->activation->batchnorm)
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`

    Examples::
        >>> conv = Conv1DBlock(in_c=3, out_c=16, kernel_sz=3)
        >>>
        >>> conv = Conv1DBlock(in_c=16, out_c=32, kernel_sz=3, stride=2)
        >>> 
        >>> conv = Conv1DBlock(in_c=3, out_c=16, kernel_sz=3, act='swish', bn=True)
    '''

    def __init__(self, in_c:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1, act:str='relu', bn:bool=False,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d):
        super().__init__()
        store_attr(but=['padding', 'kernel_sz'])
        self.pad,self.ks = padding,kernel_sz
        if self.pad == 'auto': self.pad = self.get_padding(self.ks)
        self.set_layers()
    
    @staticmethod
    def get_padding(kernel_sz:int) -> int:
        r'''
        Automatically computes the required padding to keep the number of columns equal before and after convolution

        Arguments:
            kernel_sz: width of convolutional kernel

        Returns:
            size of padding
        '''

        return kernel_sz//2
        
    def set_layers(self) -> None:
        r'''
        One of the main function to overload when inheriting from class. By default calls `self.get_conv_layer` once but can be changed to produce more
        complicated architectures. Sets `self.layers` to the constructed architecture.
        '''

        self.layers = self.get_conv_layer(in_c=self.in_c, out_c=self.out_c, kernel_sz=self.ks, padding=self.pad, stride=self.stride)
        
    def get_conv_layer(self, in_c:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1, pre_act:bool=False, groups:int=1) -> nn.Module:
        r'''
        Builds a sandwich of layers with a single concilutional layer, plus any requested batch norm and activation.
        Also initialises layers to requested scheme.

        Arguments:
            in_c: number of input channels (number of features per object / rows in input matrix)
            out_c: number of output channels (number of features / rows in output matrix)
            kernel_sz: width of kernel, i.e. the number of columns to overlay
            padding: amount of padding columns to add at start and end of convolution.
                If left as 'auto', padding will be automatically computed to conserve the number of columns.
            stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
                stride 2 = kernel centred on ever other column and input size halved, et cetera.
            pre_act: whether to apply batchnorm and activation layers prior to the weight layer, or afterwards
            groups: number of blocks of connections from input channels to output channels
        '''
        
        if padding == 'auto': padding = self.get_padding(kernel_sz)
        layers = []
        if pre_act:
            if self.bn: layers.append(self.bn_class(in_c))
            if self.act != 'linear': layers.append(self.lookup_act(self.act))
                    
        layers.append(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_sz, padding=padding, stride=stride, groups=groups))
        self.lookup_init(self.act)(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        if not pre_act:
            if self.act != 'linear': layers.append(self.lookup_act(self.act))
            if self.bn: layers.append(self.bn_class(out_c))
        return nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        r'''
        Passes input through the layers.
        Might need to be overloaded in inheritance, depending on architecture.

        Arguments:
            x: input tensor
        
        Returns:
            Resulting tensor
        '''

        return self.layers(x)


class Res1DBlock(Conv1DBlock):
    r'''
    Basic building block for a building and applying a pair of residually connected 1D convolutional layers (https://arxiv.org/abs/1512.03385).
    Batchnorm is applied 'pre-activation' as per https://arxiv.org/pdf/1603.05027.pdf, and convolutional shortcuts (again https://arxiv.org/pdf/1603.05027.pdf)
    are used when the stride of the first layer is greater than 1, or the number of input channels does not equal the number of output channels.

    Arguments:
        in_c: number of input channels (number of features per object / rows in input matrix)
        out_c: number of output channels (number of features / rows in output matrix)
        kernel_sz: width of kernel, i.e. the number of columns to overlay
        padding: amount of padding columns to add at start and end of convolution.
            If left as 'auto', padding will be automatically computed to conserve the number of columns.
        stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
            stride 2 = kernel centred on ever other column and input size halved, et cetera.
        act: string representation of argument to pass to lookup_act
        bn: whether to use batch normalisation (order is pre-activation: batchnorm->activation->weights)
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer

    Examples::
        >>> conv = Res1DBlock(in_c=16, out_c=16, kernel_sz=3)
        >>>
        >>> conv = Res1DBlock(in_c=16, out_c=32, kernel_sz=3, stride=2)
        >>> 
        >>> conv = Res1DBlock(in_c=16, out_c=16, kernel_sz=3, act='swish', bn=True)
    '''
        
    def set_layers(self):
        r'''
        Constructs a pair of pre-activation convolutional layers, and a shortcut layer if necessary.
        '''

        self.layers = nn.Sequential(self.get_conv_layer(in_c=self.in_c,  out_c=self.out_c, kernel_sz=self.ks, padding=self.pad, stride=self.stride,
                                                        pre_act=True),
                                    self.get_conv_layer(in_c=self.out_c, out_c=self.out_c, kernel_sz=self.ks, padding=self.pad, stride=1,
                                                        pre_act=True))
        if self.stride != 1 or self.in_c != self.out_c:
            self.shortcut = nn.Conv1d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=self.stride)
        else:
            self.shortcut = None

    def forward(self, x:Tensor) -> Tensor:
        r'''
        Passes input through the pair of layers and then adds the resulting tensor to the original input,
        which may be passed through a shortcut connection is necessary.

        Arguments:
            x: input tensor
        
        Returns:
            Resulting tensor
        '''

        skip = x if self.shortcut is None else self.shortcut(x)
        return skip + self.layers(x)


class ResNeXt1DBlock(Conv1DBlock):
    r'''
    Basic building block for a building and applying a set of residually connected groups of 1D convolutional layers (https://arxiv.org/abs/1611.05431).
    Batchnorm is applied 'pre-activation' as per https://arxiv.org/pdf/1603.05027.pdf, and convolutional shortcuts (again https://arxiv.org/pdf/1603.05027.pdf)
    are used when the stride of the first layer is greater than 1, or the number of input channels does not equal the number of output channels.

    Arguments:
        in_c: number of input channels (number of features per object / rows in input matrix)
        inter_c: number of intermediate channels in groups
        cardinality: number of groups
        out_c: number of output channels (number of features / rows in output matrix)
        kernel_sz: width of kernel, i.e. the number of columns to overlay
        padding: amount of padding columns to add at start and end of convolution.
            If left as 'auto', padding will be automatically computed to conserve the number of columns.
        stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
            stride 2 = kernel centred on ever other column and input size halved, et cetera.
        act: string representation of argument to pass to lookup_act
        bn: whether to use batch normalisation (order is pre-activation: batchnorm->activation->weights)
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
        bn_class: class to use for BatchNorm, default is `nn.BatchNorm1d`

    Examples::
        >>> conv = ResNeXt1DBlock(in_c=32, inter_c=4, cardinality=4, out_c=32, kernel_sz=3)
        >>>
        >>> conv = ResNeXt1DBlock(in_c=32, inter_c=4, cardinality=4, out_c=32, kernel_sz=3, stride=2)
        >>> 
        >>> conv = ResNeXt1DBlock(in_c=32, inter_c=4, cardinality=4, out_c=32, kernel_sz=3, act='swish', bn=True)
    '''

    def __init__(self, in_c:int, inter_c:int, cardinality:int, out_c:int, kernel_sz:int, padding:Union[int,str]='auto', stride:int=1, act:str='relu',
                 bn:bool=False,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, bn_class:Callable[[int],nn.Module]=nn.BatchNorm1d):
        self.inter_c,self.cardinality = inter_c,cardinality
        super().__init__(in_c=in_c, out_c=out_c, kernel_sz=kernel_sz, padding=padding, stride=stride, act=act, bn=bn,
                         lookup_init=lookup_init, lookup_act=lookup_act, bn_class=bn_class)
        
    def set_layers(self):
        r'''
        Constructs a set of grouped pre-activation convolutional layers, and a shortcut layer if necessary.
        '''

        self.layers = nn.Sequential(self.get_conv_layer(in_c=self.in_c,  out_c=self.inter_c*self.cardinality, kernel_sz=1, stride=self.stride, pre_act=True),
                                    self.get_conv_layer(in_c=self.inter_c*self.cardinality, out_c=self.inter_c*self.cardinality, kernel_sz=self.ks,
                                                        padding=self.pad, stride=1, groups=self.cardinality, pre_act=True),
                                    self.get_conv_layer(in_c=self.inter_c*self.cardinality, out_c=self.out_c, kernel_sz=1, stride=1, pre_act=True))
        if self.stride != 1 or self.in_c != self.out_c:
            self.shortcut = nn.Conv1d(in_channels=self.in_c, out_channels=self.out_c, kernel_size=1, stride=self.stride)
        else:
            self.shortcut = None

    def forward(self, x:Tensor) -> Tensor:
        r'''
        Passes input through the set of layers and then adds the resulting tensor to the original input,
        which may be passed through a shortcut connection is necessary.

        Arguments:
            x: input tensor
        
        Returns:
            Resulting tensor
        '''

        skip = x if self.shortcut is None else self.shortcut(x)
        return skip + self.layers(x)


class AdaptiveAvgMaxConcatPool1d(nn.Module):
    r'''
    Layer that reduces the size of each channel to the specified size, via two methods: average pooling and max pooling.
    The outputs are then concatenated channelwise.

    Arguments:
        sz: Requested output size, default reduces each channel to 2*1 elements.
            The first element is the maximum value in the channel and the other is the average value in the channel.
    '''

    def __init__(self, sz:Optional[Union[int,Tuple[int,...]]]=None):
        super().__init__()
        self._setup(sz)

    def _setup(self, sz:Optional[Union[int,Tuple[int]]]=None) -> None:
        if sz is None: sz = (1)
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        r'''
        Passes input through the adaptive pooling.

        Arguments:
            x: input tensor
        
        Returns:
            Resulting tensor
        '''
        
        return torch.cat([self.mp(x), self.ap(x)], 1)


class AdaptiveAvgMaxConcatPool2d(AdaptiveAvgMaxConcatPool1d):
    r'''
    Layer that reduces the size of each channel to the specified size, via two methods: average pooling and max pooling.
    The outputs are then concatenated channelwise.

    Arguments:
        sz: Requested output size, default reduces each channel to 2*1 elements.
            The first element is the maximum value in the channel and the other is the average value in the channel.
    '''

    def _setup(self, sz:Optional[Union[int,Tuple[int,int]]]=None) -> None:
        if sz is None: sz = (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
        

class AdaptiveAvgMaxConcatPool3d(AdaptiveAvgMaxConcatPool1d):
    r'''
    Layer that reduces the size of each channel to the specified size, via two methods: average pooling and max pooling.
    The outputs are then concatenated channelwise.

    Arguments:
        sz: Requested output size, default reduces each channel to 2*1 elements.
            The first element is the maximum value in the channel and the other is the average value in the channel.
    '''

    def _setup(self, sz:Optional[Union[int,Tuple[int,int,int]]]=None) -> None:
        if sz is None: sz = (1,1,1)
        self.ap = nn.AdaptiveAvgPool3d(sz)
        self.mp = nn.AdaptiveMaxPool3d(sz)


class SEBlock1d(nn.Module):
    r'''
    Squeeze-excitation block [Hu, Shen, Albanie, Sun, & Wu, 2017](https://arxiv.org/abs/1709.01507).
    Incoming data is averaged per channel, fed through a single layer of width `n_in//r` and the chose activation, then a second layer of width `n_in` and a sigmoid activation.
    Channels in the original data are then multiplied by the learned channe weights.

    Arguments:
        n_in: number of incoming channels
        r: the reduction ratio for the channel compression
        act: string representation of argument to pass to lookup_act
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
    '''

    def __init__(self, n_in:int, r:int, act:str='relu', lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act):
        super().__init__()
        self.n_in,self.r,self.act,self.lookup_init,self.lookup_act = n_in,r,act,lookup_init,lookup_act
        self.layers = self._get_layers()
        self.sz = [1]
        self.pool = nn.AdaptiveAvgPool1d(self.sz)

    def _get_layers(self) -> nn.Sequential:
        c = np.max((2,self.n_in//self.r))
        layers = [nn.Linear(self.n_in, c), self.lookup_act(self.act),
                  nn.Linear(c, self.n_in), nn.Sigmoid()]
        self.lookup_init(self.act)(layers[0].weight)
        self.lookup_init('sigmoid')(layers[2].weight)
        nn.init.zeros_(layers[0].bias)
        nn.init.zeros_(layers[2].bias)
        return nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        r'''
        Rescale the incoming tensor by the learned channel weights

        Arguments:
            x: incoming tensor

        Returns:
            x*y, where y is the output of the squeeze-excitation network 
        '''
        
        y = self.pool(x).view(-1,self.n_in)
        y = self.layers(y).view(-1,self.n_in,*self.sz)
        return x*y


class SEBlock2d(SEBlock1d):
    r'''
    Squeeze-excitation block [Hu, Shen, Albanie, Sun, & Wu, 2017](https://arxiv.org/abs/1709.01507).
    Incoming data is averaged per channel, fed through a single layer of width `n_in//r` and the chose activation, then a second layer of width `n_in` and a sigmoid activation.
    Channels in the original data are then multiplied by the learned channe weights.

    Arguments:
        n_in: number of incoming channels
        r: the reduction ratio for the channel compression
        act: string representation of argument to pass to lookup_act
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
    '''

    def __init__(self, n_in:int, r:int, act:str='relu', lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act):
        super().__init__()
        self.n_in,self.r,self.act,self.lookup_init,self.lookup_act = n_in,r,act,lookup_init,lookup_act
        self.layers = self._get_layers()
        self.sz = [1,1]
        self.pool = nn.AdaptiveAvgPool2d(self.sz)


class SEBlock3d(SEBlock1d):
    r'''
    Squeeze-excitation block [Hu, Shen, Albanie, Sun, & Wu, 2017](https://arxiv.org/abs/1709.01507).
    Incoming data is averaged per channel, fed through a single layer of width `n_in//r` and the chose activation, then a second layer of width `n_in` and a sigmoid activation.
    Channels in the original data are then multiplied by the learned channe weights.

    Arguments:
        n_in: number of incoming channels
        r: the reduction ratio for the channel compression
        act: string representation of argument to pass to lookup_act
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
    '''

    def __init__(self, n_in:int, r:int, act:str='relu', lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act):
        super().__init__()
        self.n_in,self.r,self.act,self.lookup_init,self.lookup_act = n_in,r,act,lookup_init,lookup_act
        self.layers = self._get_layers()
        self.sz = [1,1,1]
        self.pool = nn.AdaptiveAvgPool3d(self.sz)
