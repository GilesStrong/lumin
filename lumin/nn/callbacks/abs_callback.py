from abc import ABC


class AbsCallback(ABC):
    '''Abstract callback class for typing'''
    def __init__(self):                        pass
    def set_model(self, **kargs):              pass
    def set_plot_settings(self, **kargs):      pass
    def on_train_begin(self, **kargs):         pass  
    def on_train_end(self,   **kargs):         pass  
    def on_epoch_begin(self, **kargs):         pass
    def on_epoch_end(self,   **kargs):         pass
    def on_batch_begin(self, **kargs):         pass
    def on_batch_end(self,   **kargs):         pass
    def on_eval_begin(self, **kargs):          pass
    def on_eval_end(self,   **kargs):          pass
    def on_backwards_begin(self, **kargs):     pass
    def on_backwards_end(self,   **kargs):     pass
