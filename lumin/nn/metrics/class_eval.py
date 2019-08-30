import numpy as np
from typing import List

from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder
from ...evaluation.ams import ams_scan_quick, ams_scan_slow
from ...utils.misc import to_binary_class

__all__ = ['AMS', 'MultiAMS']


class AMS(EvalMetric):
    r'''
    Class to compute maximum Approximate Median Significance (https://arxiv.org/abs/1007.1727) using classifier which directly predicts the class of data in a
    binary classifiaction problem.
    AMS is computed on a single fold of data provided by a :class:`~lumin.nn.data.fold_yielder.FoldYielder` and automatically reweights data by event
    multiplicity to account missing weights.

    Arguments:
        n_total:total number of events in entire data set
        wgt_name: name of weight group in fold file to use. N.B. if you have reweighted to balance classes, be sure to use the un-reweighted weights.
        targ_name: name of target group in fold file
        br: constant bias offset for background yield
        syst_unc_b: fractional systematic uncertainty on background yield
        use_quick_scan: whether to optimise AMS by the :meth:`~lumin.evaluation.ams.ams_scan_quick` method (fast but suffers floating point precision)
            if False use :meth:`~lumin.evaluation.ams.ams_scan_slow` (slower but more accurate)

    Examples::
        >>> ams_metric = AMS(n_total=250000, br=10, wgt_name='gen_orig_weight')
        >>>
        >>> ams_metric = AMS(n_total=250000, syst_unc_b=0.1,
        ...                  wgt_name='gen_orig_weight', use_quick_scan=False)
    '''

    def __init__(self, n_total:int, wgt_name:str, targ_name:str='targets', br:float=0, syst_unc_b:float=0, use_quick_scan:bool=True):
        super().__init__(targ_name=targ_name, wgt_name=wgt_name)
        self.n_total,self.br,self.syst_unc_b,self.use_quick_scan,self.lower_metric_better = n_total,br,syst_unc_b,use_quick_scan,False

    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        r'''
        Compute maximum AMS on fold using provided predictions.

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            Maximum AMS computed on reweighted data from fold

        Examples::
            >>> ams = ams_metric.evaluate(train_fy, val_id, val_preds)
        '''

        df = self.get_df(fy, idx, y_pred)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams


class MultiAMS(AMS):
    r'''
    Class to compute maximum Approximate Median Significance (https://arxiv.org/abs/1007.1727) using classifier which predicts the class of data in a multiclass
    classifiaction problem which can be reduced to a binary classification problem
    AMS is computed on a single fold of data provided by a :class:`~lumin.nn.data.fold_yielder.FoldYielder` and automatically reweights data by event
    multiplicity to account missing weights.

    Arguments:
        n_total:total number of events in entire data set
        wgt_name: name of weight group in fold file to use. N.B. if you have reweighted to balance classes, be sure to use the un-reweighted weights.
        targ_name: name of target group in fold file which indicates whether the event is signal or background
        zero_preds: list of predicted classes which correspond to class 0 in the form pred_[i], where i is a NN output index
        one_preds: list of predicted classes which correspond to class 1 in the form pred_[i], where i is a NN output index 
        br: constant bias offset for background yield
        syst_unc_b: fractional systematic uncertainty on background yield
        use_quick_scan: whether to optimise AMS by the :meth:`~lumin.evaluation.ams.ams_scan_quick` method (fast but suffers floating point precision)
            if False use :meth:`~lumin.evaluation.ams.ams_scan_slow` (slower but more accurate)

    Examples::
        >>> ams_metric = MultiAMS(n_total=250000, br=10, targ_name='gen_target',
        ...                       wgt_name='gen_orig_weight',
        ...                       zero_preds=['pred_0', 'pred_1', 'pred_2'],
        ...                       one_preds=['pred_3'])
        >>>
        >>> ams_metric = MultiAMS(n_total=250000, syst_unc_b=0.1,
        ...                       targ_name='gen_target',
        ...                       wgt_name='gen_orig_weight',
        ...                       use_quick_scan=False,
        ...                       zero_preds=['pred_0', 'pred_1', 'pred_2'],
        ...                       one_preds=['pred_3'])
    '''

    def __init__(self, n_total:int, wgt_name:str, targ_name:str, zero_preds:List[str], one_preds:List[str], br:float=0, syst_unc_b:float=0,
                 use_quick_scan:bool=True):
        super().__init__(n_total=n_total, br=br, syst_unc_b=syst_unc_b, use_quick_scan=use_quick_scan, targ_name=targ_name, wgt_name=wgt_name)
        self.zero_preds,self.one_preds = zero_preds,one_preds

    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        r'''
        Compute maximum AMS on fold using provided predictions.

        Arguments:
            fy: :class:`~lumin.nn.data.fold_yielder.FoldYielder` interfacing to data
            idx: fold index corresponding to fold for which y_pred was computed
            y_pred: predictions for fold

        Returns:
            Maximum AMS computed on reweighted data from fold

        Examples::
            >>> ams = ams_metric.evaluate(train_fy, val_id, val_preds)
        '''

        # TODO: make the zero and one preds more user-friendly
        
        df = self.get_df(fy, idx, y_pred)
        to_binary_class(df, self.zero_preds, self.one_preds)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams
