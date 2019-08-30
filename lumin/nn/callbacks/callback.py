from typing import Optional

from .abs_callback import AbsCallback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings

__all__ = ['Callback']


class Callback(AbsCallback):
    r'''
    Base callback class from which other callbacks should inherit.

    Arguments:
        model: model to refer to during training
        plot_settings: PlotSettings class
    '''

    def __init__(self, model:Optional[AbsModel]=None, plot_settings:PlotSettings=PlotSettings()):
        if model is not None: self.set_model(model)
        self.set_plot_settings(plot_settings)

    def set_model(self, model:AbsModel) -> None:
        r'''
        Sets the callback's model in order to allow the callback to access and adjust model parameters

        Arguments:
            model: model to refer to during training
        '''
        
        self.model = model
    
    def set_plot_settings(self, plot_settings:PlotSettings) -> None:
        r'''
        Sets the plot settings for any plots produced by the callback

        Arguments:
            plot_settings: PlotSettings class
        '''

        self.plot_settings = plot_settings
        