from typing import List, Optional
from sklearn.metrics import accuracy_score, roc_auc_score
from fastcore.all import store_attr

from .eval_metric import EvalMetric
from ...evaluation.ams import ams_scan_quick, ams_scan_slow
from ...utils.misc import to_binary_class

__all__ = ['AMS', 'MultiAMS', 'BinaryAccuracy', 'RocAucScore']


class AMS(EvalMetric):
    r'''
    Class to compute maximum Approximate Median Significance (https://arxiv.org/abs/1007.1727) using classifier which directly predicts the class of data in a
    binary classifiaction problem.
    AMS is computed on a single fold of data provided by a :class:`~lumin.nn.data.fold_yielder.FoldYielder` and automatically reweights data by event
    multiplicity to account missing weights.

    Arguments:
        n_total:total number of events in entire data set
        wgt_name: name of weight group in fold file to use. N.B. if you have reweighted to balance classes, be sure to use the un-reweighted weights.
        br: constant bias offset for background yield
        syst_unc_b: fractional systematic uncertainty on background yield
        use_quick_scan: whether to optimise AMS by the :meth:`~lumin.evaluation.ams.ams_scan_quick` method (fast but suffers floating point precision)
            if False use :meth:`~lumin.evaluation.ams.ams_scan_slow` (slower but more accurate)
        name: optional name for metric, otherwise will be 'AMS'
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted

    Examples::
        >>> ams_metric = AMS(n_total=250000, br=10, wgt_name='gen_orig_weight')
        >>>
        >>> ams_metric = AMS(n_total=250000, syst_unc_b=0.1,
        ...                  wgt_name='gen_orig_weight', use_quick_scan=False)

    '''

    def __init__(self, n_total:int, wgt_name:str, br:float=0, syst_unc_b:float=0, use_quick_scan:bool=True, name:Optional[str]='AMS', main_metric:bool=True):
        super().__init__(name=name, lower_metric_better=False, main_metric=main_metric)
        store_attr(but=['name', 'main_metric'])

    def evaluate(self) -> float:
        r'''
        Compute maximum AMS on fold using provided predictions.

        Returns:
            Maximum AMS computed on reweighted data from fold
        '''

        df = self.get_df()
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(df), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(df), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams


class MultiAMS(EvalMetric):
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
        name: optional name for metric, otherwise will be 'AMS'
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted

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
                 use_quick_scan:bool=True, name:Optional[str]='AMS', main_metric:bool=True):
        super().__init__(name=name, lower_metric_better=False, main_metric=main_metric)
        store_attr(but=['name', 'main_metric'])

    def evaluate(self) -> float:
        r'''
        Compute maximum AMS on fold using provided predictions.

        Returns:
            Maximum AMS computed on reweighted data from fold
        '''

        # TODO: make the zero and one preds more user-friendly
        
        df = self.get_df()
        to_binary_class(df, self.zero_preds, self.one_preds)
        if self.use_quick_scan: ams, _ = ams_scan_quick(df, wgt_factor=self.n_total/len(df), br=self.br, syst_unc_b=self.syst_unc_b)
        else:                   ams, _ = ams_scan_slow(df,  wgt_factor=self.n_total/len(df), br=self.br, syst_unc_b=self.syst_unc_b, show_prog=False)
        return ams


class BinaryAccuracy(EvalMetric):
    r'''
    Computes and returns the accuracy of a single-output model for binary classification tasks.

    Arguments:
        threshold: minimum value of model prediction that will be considered a prediction of class 1. Values below this threshold will be considered predictions
            of class 0. Default = 0.5.
        name: optional name for metric, otherwise will be 'Acc'
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted

    Examples::
        >>> acc_metric = BinaryAccuracy()
        >>>
        >>> acc_metric = BinaryAccuracy(threshold=0.8)
    '''
    
    def __init__(self, threshold:float=0.5, name:Optional[str]='Acc', main_metric:bool=True):
        super().__init__(name=name, lower_metric_better=False, main_metric=main_metric)
        store_attr(but=['name', 'main_metric'])

    def evaluate(self) -> float:
        r'''
        Computes the (weighted) accuracy for a set of targets and predictions for a given threshold.

        Returns:
            The (weighted) accuracy for the specified threshold
        '''

        self.preds = (self.preds >= self.threshold).astype(int)
        return accuracy_score(self.targets, y_pred=self.preds, sample_weight=self.weights)
   

class RocAucScore(EvalMetric):
    r'''
    Computes and returns the area under the Receiver Operator Characteristic curve (ROC AUC) of a classifier model.

    Arguments:
        average: As per scikit-learn. {'micro', 'macro', 'samples', 'weighted'} or None, default='macro'
            If ``None``, the scores for each class are returned. Otherwise,
            this determines the type of averaging performed on the data:
            Note: multiclass ROC AUC currently only handles the 'macro' and
            'weighted' averages.

            ``'micro'``:
                Calculate metrics globally by considering each element of the label
                indicator matrix as a label.

            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.

            ``'weighted'``:
                Calculate metrics for each label, and find their average, weighted
                by support (the number of true instances for each label).

            ``'samples'``:
                Calculate metrics for each instance, and find their average.

            Will be ignored when ``y_true`` is binary.
        max_fpr: As per scikit-learn. float > 0 and <= 1, default=None
            If not ``None``, the standardized partial AUC over the range
            [0, max_fpr] is returned. For the multiclass case, ``max_fpr``,
            should be either equal to ``None`` or ``1.0`` as AUC ROC partial
            computation currently is not supported for multiclass.
        multi_class:  As per scikit-learn. {'raise', 'ovr', 'ovo'}, default='raise'
            Multiclass only. Determines the type of configuration to use. The
            default value raises an error, so either ``'ovr'`` or ``'ovo'`` must be
            passed explicitly.

            ``'ovr'``:
                Computes the AUC of each class against the rest. This
                treats the multiclass case in the same way as the multilabel case.
                Sensitive to class imbalance even when ``average == 'macro'``,
                because class imbalance affects the composition of each of the
                'rest' groupings.

            ``'ovo'``:
                Computes the average AUC of all possible pairwise combinations of
                classes. Insensitive to class imbalance when
                ``average == 'macro'``.
        name: optional name for metric, otherwise will be 'Acc'
        main_metric: whether this metic should be treated as the primary metric for SaveBest and EarlyStopping
            Will automatically set the first EvalMetric to be main if multiple primary metrics are submitted

    Examples::
        >>> auc_metric = RocAucScore()
        >>>
        >>> auc_metric = RocAucScore(max_fpr=0.2)
        >>>
        >>> auc_metric = RocAucScore(multi_class='ovo')
    '''
    
    def __init__(self, average:Optional[str]='macro', max_fpr:Optional[float]=None, multi_class:str='raise', name:Optional[str]='ROC AUC',
                 main_metric:bool=True):
        super().__init__(name=name, lower_metric_better=False, main_metric=main_metric)
        store_attr(but=['name', 'main_metric'])

    def evaluate(self) -> float:
        r'''
        Computes the (weighted) (averaged) ROC AUC for a set of targets and predictions.

        Returns:
            The (weighted) (averaged) ROC AUC for the specified threshold
        '''

        return roc_auc_score(y_true=self.targets, y_score=self.preds, sample_weight=self.weights,
                             average=self.average, max_fpr=self.max_fpr, multi_class=self.multi_class)
