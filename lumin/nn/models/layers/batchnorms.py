from torch import nn, Tensor

__all__ = ['LCBatchNorm1d']


class LCBatchNorm1d(nn.Module):
    r'''
    Wrapper class for 1D batchnorm to make it run over (Batch x length x channel) data for use in NNs designed to be broadcast across matrix data.

    Arguments:
        bn: base 1D batchnorm module to call
    '''

    def __init__(self, bn:nn.BatchNorm1d):
        super().__init__()
        self.bn = bn
    
    def forward(self, x:Tensor) -> Tensor: return self.bn(x.transpose(-1,-2)).transpose(-1,-2)
