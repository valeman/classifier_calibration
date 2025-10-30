import logging
import pandas as pd
import numpy as np


lg = logging.getLogger(__name__)


class PreProcessing:
    """
    The PreProcessing class wraps around each pre processing technique to provide a standard API.
    Only supports (X,Y) to (X,Y) maps.
    """

    def __init__(self):
        raise NotImplementedError

    def apply(self, x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError
        return x, y


class PostProcessing:
    """
    The PostProcessing class wraps around each post processing technique to provide a standard API.
    Only supports Y to Y maps.
    """

    def __init__(self, pp_name, pp_class, instatiator_fn, fit_fn, predict_prob_fn):
        self.pp_name = pp_name
        self.pp_class = pp_class
        self.pp = None
        self.instatiator_fn = instatiator_fn
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn

    def instantiate(self, meta_data: dict) -> None:
        lg.info(
            f"Post-processing:{self.pp_name} instantiated with: \n{self.instatiator_fn(meta_data)}"
        )
        self.pp = self.pp_class(**self.instatiator_fn(meta_data))

    def fit(self, y_instance: np.array, y_target: pd.Series) -> None:
        self._fit_fn(self.pp, y_instance, y_target)

    def predict_prob(self, y: np.array) -> np.array:
        return np.asarray(self._predict_prob_fn(self.pp, y))


class Learner:
    """
    The learner class wraps around each learner to provide a standard API.
    """

    def __init__(
        self,
        learner_name,
        learner_class,
        instatiator_fn,
        fit_fn,
        predict_prob_fn,
        pre_trained=False,
    ):
        self.learner_name = learner_name
        self.learner_class = learner_class
        self.learner = None
        self.instatiator_fn = instatiator_fn
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn
        self.pre_trained = pre_trained

    def instantiate(self, meta_data: dict) -> None:
        lg.info(
            f"Learner:{self.learner_name} instantiated with:\n{self.instatiator_fn(meta_data)}"
        )
        self.learner = self.learner_class(**self.instatiator_fn(meta_data))

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self._fit_fn(self.learner, x, y)

    def predict_prob(self, x: pd.DataFrame) -> np.array:
        return np.asarray(self._predict_prob_fn(self.learner, x))
