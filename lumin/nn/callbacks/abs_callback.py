from abc import ABC


__all__ = []


class OldAbsCallback(ABC):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.abs_callback.AbsCallback`.
        It is a copy of the old `AbsCallback` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self):                    pass
    def set_model(self, **kargs):          pass
    def set_plot_settings(self, **kargs):  pass
    def on_train_begin(self, **kargs):     pass  
    def on_train_end(self,   **kargs):     pass  
    def on_epoch_begin(self, **kargs):     pass
    def on_epoch_end(self,   **kargs):     pass
    def on_batch_begin(self, **kargs):     pass
    def on_batch_end(self,   **kargs):     pass
    def on_eval_begin(self, **kargs):      pass
    def on_eval_end(self,   **kargs):      pass
    def on_backwards_begin(self, **kargs): pass
    def on_backwards_end(self,   **kargs): pass
    def on_pred_begin(self, **kargs):      pass
    def on_pred_end(self,   **kargs):      pass
    

class AbsCallback(ABC):
    r'''
    Abstract callback passing though all action points and indicating where callbacks can affect the model.
    See :meth:`~lumin.nn.models.model.Model.fit` and :meth:`~lumin.nn.models.model.Model.predict_by` to see where exactly these action points are called.
    '''

    def __init__(self): pass
    def set_model(self) -> None: pass
    def set_plot_settings(self): pass

    def on_train_begin(self) -> None: pass
    def on_train_end(self) -> None:   pass

    def on_epoch_begin(self) -> None: pass
    def on_epoch_end(self) -> None:   pass

    def on_fold_begin(self) -> None: pass
    def on_fold_end(self) -> None:   pass

    def on_batch_begin(self) -> None: pass
    def on_batch_end(self) -> None:   pass

    def on_forwards_end(self) -> None: pass

    def on_backwards_begin(self) -> None: pass
    def on_backwards_end(self) -> None:   pass

    def on_pred_begin(self) -> None: pass
    def on_pred_end(self) -> None:   pass
