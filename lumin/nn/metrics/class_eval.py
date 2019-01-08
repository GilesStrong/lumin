import numpy as np

from .eval_metric import EvalMetric
from ..data.fold_yielder import FoldYielder
from ...evaluation.ams import ams_scan_quick, ams_scan_slow


class AMS(EvalMetric):
    def __init__(self, n_total:int, br:float=0, syst_unc_b:float=0, use_quick_scan:bool=True,
                 targ_name:str='targets', weight_name:str='orig_weights'):
        super().__init__(targ_name, weight_name)
        self.n_total,self.br,self.syst_unc_b,self.use_quick_scan = n_total,br,syst_unc_b,use_quick_scan

    def evaluate(self, data:FoldYielder, index:int, y_pred:np.ndarray) -> float:
        df = self.get_df(data, index, y_pred)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, w_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  w_factor=self.n_total/len(y_pred), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams
