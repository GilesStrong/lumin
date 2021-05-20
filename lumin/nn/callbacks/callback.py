from .abs_callback import AbsCallback
from ..models.abs_model import AbsModel
from ...plotting.plot_settings import PlotSettings

__all__ = ['Callback']


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
    
    def on_pred_begin(self) -> None:
        if self.model is None:
            raise AttributeError(f"The model for {type(self).__name__} callback has not been set. Please call set_wrapper before on_model_begin.")
