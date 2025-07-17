from pycaleva import CalibrationEvaluator
from calfram.calibration_framework import CalibrationFramework
from components.utils import all_numbers_and_finite
import sklearn.metrics as skm 
import pandas as pd
import numpy as np


class PerformanceMeasures:
    """
    The PerformanceMeasures class collects a set of performance measurements and calculates them on demand. 
    """

    def __init__(self, suite_name:str):
        self.suite_name = suite_name
        self.measure_functions = {}

        match suite_name:
            case "v.1":
                self.init_v1()
            case _:
                raise NotImplementedError
        
    def calc_perf(self, x_test:pd.DataFrame, y_prob:np.ndarray ,y_pred:np.ndarray, y_test:np.ndarray) -> dict:
        """
            Calcualtes and returns all performance measures in the suite.
            Computational measures are handled seperately. 

        Args:
            x_test (pd.DataFrame): The instances belonging to y_test
            y_prob (np.ndarray): | shape: (n, c) if binary then shape: (n, 1)  
            y_pred (np.ndarray): | shape: (n, 1)
            y_test (np.ndarray): | shape: (n, 1)

        Returns:
            dict: Key: name of performance measure, Value: The calculated value of the measure
        """
        self.qc_binary_input(x_test, y_prob, y_pred, y_test)
        perf_measures = {}
        for measure,function in self.measure_functions.items():
            perf_measures[measure] = function(x_test, y_prob ,y_pred, y_test)
        return perf_measures
    

    def init_v1(self) -> None:
        """
        Collects all the functions which calculates the below measures into measure_functions: 
            Brier score, Spiegelhalter Z statistic, Log-loss, ECE frequency, ECI global,
            ECI balance, AUC-ROC, Accuracy, Recall ,Precision and F1 Score

        x_test:pd.DataFrame
        y_....:np.ndarray | shape: (n,)
        
        x_test is a pandas dataframe where the instances belong to the labels y_test.
        y_prob is the probability of class 1 in the intervall [0,1].
        y_pred is the predicted class, 0 or 1.
        """
        
        ce = lambda y_test, y_prob: CalibrationEvaluator(y_test, y_prob, outsample=True)
        
        self.measure_functions["brier_score"] = lambda                x_test, y_prob ,y_pred, y_test:  ce(y_test, y_prob).brier
        self.measure_functions["spiegelhalter_z_statistic"] = lambda  x_test, y_prob ,y_pred, y_test:  np.clip(ce(y_test, y_prob).z_test().statistic, a_min=-12, a_max=12)
        self.measure_functions["log_loss"] = lambda                   x_test, y_prob ,y_pred, y_test:  skm.log_loss(y_test, y_prob, normalize=True)

        self.measure_functions["auc_roc"] = lambda    x_test, y_prob ,y_pred, y_test:   ce(y_test, y_prob).auroc

        self.measure_functions["accuracy"] = lambda   x_test, y_prob ,y_pred, y_test:   skm.accuracy_score(y_test, y_pred, normalize=True) 
        self.measure_functions["recall_1"] = lambda     x_test, y_prob ,y_pred, y_test:   skm.recall_score(y_test, y_pred,pos_label=1, average='binary')
        self.measure_functions["recall_0"] = lambda     x_test, y_prob ,y_pred, y_test:   skm.recall_score(y_test, y_pred,pos_label=0, average='binary')
        self.measure_functions["f1_score"] = lambda   x_test, y_prob ,y_pred, y_test:   skm.f1_score(y_test, y_pred, pos_label=1, average='binary')
        self.measure_functions["precision"] = lambda  x_test, y_prob ,y_pred, y_test:   skm.precision_score(y_test, y_pred, pos_label=1, average='binary')
        
        def calfram_measures(measure_name, y_test, y_prob, y_pred):
            #Convert shape: (n,) to shape: (n, 2)
            if y_prob.ndim == 1:
                y_prob = y_prob.reshape(-1, 1)
                y_prob = np.concatenate([1 - y_prob, y_prob], axis=1)

            caf = CalibrationFramework()
            classes_scores = caf.select_probability(y_test # shape: (n,)
                                                    ,y_prob # shape: (n, c), where c is the number of classes 
                                                    ,y_pred # shape: (n,)
            )
        
            measures, _ = caf.calibrationdiagnosis(classes_scores, adaptive=True #automatic monotonic sweep method
            )
            class_wise_metrics = caf.classwise_calibration(measures)
            measure = None
            match measure_name:
                case "eci_global":
                    measure = class_wise_metrics["ec_g"] # ECI_global
                case "eci_balance":
                    measure = class_wise_metrics["ec_dir"] # ECI_balance
                case "ece_freq":
                     measure = class_wise_metrics["ece_freq"] #ECE_frequency 
            return measure
        
        self.measure_functions["eci_global"] = lambda     x_test, y_prob ,y_pred, y_test: calfram_measures("eci_global", y_test, y_prob, y_pred)
        self.measure_functions["eci_balance"] = lambda    x_test, y_prob ,y_pred, y_test: calfram_measures("eci_balance", y_test, y_prob, y_pred)
        self.measure_functions["ece_freq"] = lambda       x_test, y_prob ,y_pred, y_test: calfram_measures("ece_freq", y_test, y_prob, y_pred)
    
    
    def qc_binary_input(self,
                        x_test: pd.DataFrame,
                        y_prob: np.ndarray,
                        y_pred: np.ndarray,
                        y_test: np.ndarray) -> None:
        """
        Quality check inputs for a binary classifier.

        Checks:
        1. All inputs have the same length n.
        2. y_prob, y_pred, y_test are 1-D arrays of length n with numbers.
        3. y_prob values are all in [0, 1].
        4. y_pred and y_test values are only 0 or 1.

        Raises:
        TypeError:   if any of y_* isn't a 1-D numpy array or x is not a pandas dataframe.
        ValueError:  if lengths mismatch or values fall outside allowed ranges.
        """
        # 1) + 2) Length and type checks
        n = len(x_test)
        if not isinstance(x_test, pd.DataFrame):
            raise TypeError(f"x_test must be a pandas DataFrame, got {type(x_test)}")

        for name, arr in (("y_prob", y_prob), ("y_pred", y_pred), ("y_test", y_test)):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray, got {type(arr)}")
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, but has shape {arr.shape}")
            if arr.shape[0] != n:
                raise ValueError(f"Length mismatch: len(x_test)={n} but {name}.shape[0]={arr.shape[0]}")
            if not all_numbers_and_finite(arr):
                raise TypeError(f"All elements in {name} must be integers or floats")

        # 3) Probability range check
        if not np.all((y_prob >= 0) & (y_prob <= 1)):
            bad_idx = np.where((y_prob < 0) | (y_prob > 1))[0]
            raise ValueError(f"y_prob contains values outside [0,1] at indices {bad_idx.tolist()}")

        # 4) Binary‐label checks
        for name, arr in (("y_pred", y_pred), ("y_test", y_test)):
            unique_vals = np.unique(arr)
            bad = set(unique_vals) - {0, 1}
            if bad:
                raise ValueError(f"{name} contains non‐binary values {bad}; only {{0,1}} allowed")

