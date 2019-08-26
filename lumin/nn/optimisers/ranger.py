
import math
import torch
from torch.optim.optimizer import Optimizer
import itertools as it

r'''
File copied from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer which is distibuted under Apache 2.0 licence
https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/LICENSE 

Comment style adjusted and __repr__ added by Giles C Strong

If used, please cite:
Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Jiawei Han. "On the Variance of the Adaptive Learning Rate and Beyond."
arXiv preprint arXiv:1908.03265 (2019).

@article{liu2019radam,
  title={On the Variance of the Adaptive Learning Rate and Beyond},
  author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
  journal={arXiv preprint arXiv:1908.03265},
  year={2019}
}

and 

Michael R. Zhang, James Lucas, Geoffrey E. Hinton, and Jimmy Ba. "Lookahead Optimizer: k steps forward, 1 step back", CoRR (2019), arXiv:1907.08610

@article{DBLP:journals/corr/abs-1907-08610,
  author    = {Michael R. Zhang and
               James Lucas and
               Geoffrey E. Hinton and
               Jimmy Ba},
  title     = {Lookahead Optimizer: k steps forward, 1 step back},
  journal   = {CoRR},
  volume    = {abs/1907.08610},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.08610},
  archivePrefix = {arXiv},
  eprint    = {1907.08610},
  timestamp = {Tue, 23 Jul 2019 10:54:22 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1907-08610},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
'''

# credit - Lookahead implementation from LonePatient - https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
# credit2 - RAdam code by https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py


class Ranger(Optimizer):
    
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(.9,0.999), eps=1e-8, weight_decay=0):
        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')
        
        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)
        
        # now we can get to work...
        for group in self.param_groups:
            group["step_counter"] = 0
            # print("group step counter init")
                      
        # look ahead params
        self.alpha = alpha
        self.k = k 
        
        # radam buffer for state
        self.radam_buffer = [[None,None,None] for ind in range(10)]
        
        # lookahead weights
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]
        
        # don't use grad for lookahead weights
        for w in it.chain(*self.slow_weights):
            w.requires_grad = False
        
    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def __repr__(self) -> str:
        rep = super().__repr__()
        rep += f"\nk: {self.k}\nalpha: {self.alpha}"
        return rep
         
    def step(self, closure=None):
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.  
        # Uncomment if you need to use the actual closure...
        
        # if closure is not None: loss = closure()
            
        # ------------ radam
        for group in self.param_groups:
    
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
    
                p_data_fp32 = p.data.float()
    
                state = self.state[p]
    
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)
    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
    
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
    
                state['step'] += 1
                buffered = self.radam_buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size
    
                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
    
                if N_sma > 5:                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)
    
                p.data.copy_(p_data_fp32)
        
        # ---------------- end radam step
        
        # look ahead tracking and updating if latest batch = k
        for group,slow_weights in zip(self.param_groups,self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p,q in zip(group['params'],slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha,p.data - q.data)
                p.data.copy_(q.data)
            
        return loss

