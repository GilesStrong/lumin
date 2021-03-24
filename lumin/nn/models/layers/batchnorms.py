from torch import nn, Tensor


__all__ = ['LCBatchNorm1d']


class LCBatchNorm1d(nn.BatchNorm1d):
    r'''
    Version of nn.BatchNorm1d that runs over (Batch x length x channel) data for use in NNs designed to be broadcast across matrix data
    '''
    
    def forward(self, x:Tensor) -> Tensor: return super().forward(x.transpose(-1,-2)).transpose(-1,-2)
