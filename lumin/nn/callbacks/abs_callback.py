from typing import Dict, Any
from abc import ABC


class AbsCallback(ABC):
    '''Abstract callback class for typing'''
    def __init__(self): pass
    def set_model(self): pass
    def set_plot_settings(self): pass
    def on_train_begin(self, logs:Dict[str,Any]={}): pass  
    def on_train_end(self,   logs:Dict[str,Any]={}): pass  
    def on_epoch_begin(self, logs:Dict[str,Any]={}): pass
    def on_epoch_end(self,   logs:Dict[str,Any]={}): pass
    def on_batch_begin(self, logs:Dict[str,Any]={}): pass
    def on_batch_end(self,   logs:Dict[str,Any]={}): pass
