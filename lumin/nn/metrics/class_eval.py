import numpy as np
from typing import List

from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder
from ...evaluation.ams import ams_scan_quick, ams_scan_slow
from ...utils.misc import to_binary_class


class AMS(EvalMetric):
    '''Compute optimal AMS of classifier which directly predicts the class of data in a binary classifiaction problem'''
    def __init__(self, n_total:int, br:float=0, syst_unc_b:float=0, use_quick_scan:bool=True,
                 targ_name:str='targets', wgt_name:str='orig_weights'):
        super().__init__(targ_name, wgt_name)
        self.n_total,self.br,self.syst_unc_b,self.use_quick_scan,self.lower_metric_better = n_total,br,syst_unc_b,use_quick_scan,False

    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        df = self.get_df(fy, idx, y_pred)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams


class MultiAMS(AMS):
    '''Compute optimal AMS of classifier which predicts the class of data
     in a multiclass classifiaction problem which can be reduced to a binary classification problem'''
    def __init__(self, n_total:int, zero_preds:List[str], one_preds:List[str], br:float=0, syst_unc_b:float=0, use_quick_scan:bool=True,
                 targ_name:str='orig_targets', wgt_name:str='orig_weights', ):
        super().__init__(n_total, br, syst_unc_b, use_quick_scan, targ_name, wgt_name)
        self.zero_preds,self.one_preds = zero_preds,one_preds

    def evaluate(self, fy:FoldYielder, idx:int, y_pred:np.ndarray) -> float:
        df = self.get_df(fy, idx, y_pred)
        to_binary_class(df, self.zero_preds, self.one_preds)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams
