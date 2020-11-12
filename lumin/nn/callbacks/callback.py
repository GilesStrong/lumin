from typing import Optional

from .abs_callback import AbsCallback, OldAbsCallback
from ..models.abs_model import AbsModel, OldAbsModel
from ...plotting.plot_settings import PlotSettings

__all__ = ['Callback']


class OldCallback(OldAbsCallback):
    r'''
    .. Attention:: This class is depreciated in favour of :class:`~lumin.nn.callbacks.callback.Callback`.
        It is a copy of the old `Callback` class used in lumin<=0.6.
        It will be removed in V0.8
    '''

    # XXX remove in V0.8

    def __init__(self, model:Optional[AbsModel]=None, plot_settings:PlotSettings=PlotSettings()):
        if model is not None: self.set_model(model)
        self.set_plot_settings(plot_settings)

    def set_model(self, model:OldAbsModel) -> None:
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


class Callback(AbsCallback):
    r'''
    Base callback class from which other callbacks should inherit.
    '''

    def __init__(self): self.model,self.plot_settings = None,PlotSettings()
    
    def on_train_begin(self) -> None:
        if self.model is None:
            raise AttributeError(f"The model for {type(self).__name__} callback has not been set. Please call set_model before on_train_begin.")

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
        