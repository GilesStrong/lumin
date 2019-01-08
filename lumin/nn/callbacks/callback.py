from typing import Optional
from .abs_callback import AbsCallback
from ..models.model import Model


class Callback(AbsCallback):
    def __init__(self, model:Optional[Model]=None):
        if model is not None: self.set_model(model)

    def set_model(self, model:Model) -> None:
        self.model = model
        