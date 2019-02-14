from typing import Optional

from .abs_callback import AbsCallback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings


class Callback(AbsCallback):
    '''Base callback class'''
    def __init__(self, model:Optional[AbsModel]=None, plot_settings:PlotSettings=PlotSettings()):
        if model is not None: self.set_model(model)
        self.set_plot_settings(plot_settings)

    def set_model(self, model:AbsModel) -> None: self.model = model
    
    def set_plot_settings(self, plot_settings:PlotSettings) -> None: self.plot_settings = plot_settings
        