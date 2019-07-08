from pathlib import Path


class PlotSettings:
    r'''
    Class to provide control over plot appearances. Default parameters are set automatically, and can be adjusted by passing values as keyword arguments during 
    initialisation (or changed after instantiation)
    
    Arguments:
        keyword arguments: used to set relevant plotting parameters
    '''

    def __init__(self, **kargs):
        self.style       = 'whitegrid' if 'style'       not in kargs else kargs['style']
        self.cat_palette = 'tab10'     if 'cat_palette' not in kargs else kargs['cat_palette']
        self.div_palette = 'RdBu_r'    if 'div_palette' not in kargs else kargs['div_palette']
        self.seq_palette = 'viridis'   if 'seq_palette' not in kargs else kargs['seq_palette']
        
        self.tk_sz   = 16      if 'tk_sz'   not in kargs else kargs['tk_sz']
        self.tk_col  = 'black' if 'tk_col'  not in kargs else kargs['tk_col']
        self.lbl_sz  = 24      if 'lbl_sz'  not in kargs else kargs['lbl_sz']
        self.lbl_col = 'black' if 'lbl_col' not in kargs else kargs['lbl_col']

        self.title     = ''      if 'title'     not in kargs else kargs['title']
        self.title_sz  = 26      if 'title_sz'  not in kargs else kargs['title_sz']
        self.title_col = 'black' if 'title_col' not in kargs else kargs['title_col']
        self.title_loc = 'left'  if 'title_loc' not in kargs else kargs['title_loc']

        self.leg_sz  = 16     if 'leg_sz'  not in kargs else kargs['leg_sz']
        self.leg_loc = 'best' if 'leg_loc' not in kargs else kargs['leg_loc']

        self.savepath = Path('.') if 'savepath' not in kargs else kargs['savepath']
        self.format   = '.pdf'    if 'format'   not in kargs else kargs['format']
        if '.' not in self.format: self.format = '.' + self.format

        self.h_small = 4  if 'h_small' not in kargs else kargs['h_small']
        self.h_mid   = 8  if 'h_mid'   not in kargs else kargs['h_mid']
        self.h_large = 12 if 'h_large' not in kargs else kargs['h_large']
        self.h_huge  = 16 if 'h_huge'  not in kargs else kargs['h_huge']

        self.aspect = 16/9 if 'aspect' not in kargs else kargs['aspect']

        self.w_small = self.aspect*self.h_small if 'w_small' not in kargs else kargs['w_small']
        self.w_mid   = self.aspect*self.h_mid   if 'w_mid'   not in kargs else kargs['w_mid']
        self.w_large = self.aspect*self.h_large if 'w_large' not in kargs else kargs['w_large']
        self.w_huge  = self.aspect*self.h_huge  if 'w_huge'  not in kargs else kargs['w_huge']

        self.targ2class = {0: 'Background', 1: 'Signal'} if 'targ2class' not in kargs else kargs['targ2class']
        self.sample2col = None if 'sample2col' not in kargs else kargs['sample2col']

    def str2sz(self, sz:str, ax=str) -> float:
        r'''
        Used to map requested plot sizes to actual dimensions

        Arguments:
            sz: string representation of size
            ax: axis dimension requested

        Returns:
            width of plot dimension
        '''

        sz,ax = sz.lower(),ax.lower()
        if sz == 'small': return self.w_small if ax == 'x' else self.h_small
        if sz == 'mid':   return self.w_mid   if ax == 'x' else self.h_mid
        if sz == 'large': return self.w_large if ax == 'x' else self.h_large
        if sz == 'huge':  return self.w_huge  if ax == 'x' else self.h_huge

