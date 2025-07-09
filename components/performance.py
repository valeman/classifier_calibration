import components.config as config 
import sklearn.metrics as skm 
import pandas as pd
import numpy as np
from pycaleva import CalibrationEvaluator
from calfram.calibration_framework import CalibrationFramework

class PerformanceMeasures:
    def __init__(self, suite_name:str):
        
        self.suite_name = suite_name
        self.measure_functions = {}
        self.computational_cost = False

        match suite_name:
            case "v.1":
                self.computational_cost = True
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
        
        perf_measures = {}
        for key,value in self.measure_functions.items():
            perf_measures[key] = value(x_test, y_prob ,y_pred, y_test)
        return perf_measures
    

    def init_v1(self) -> None:
        """
        Collects all the functions which calculate the measures into the dictionary attribute measure_functions: 
            Brier score, Spiegelhalter Z statistic, Log-loss, ECE frequency, ECI global, ECI balance,
            AUC-ROC, Accuracy, Recall ,Precision and F1 Score

        ECE and ECI are binned with the automatic monotonic sweep method.
        
        Y labels are in {0,1}

        x_test:pd.DataFrame
        y_....:np.ndarray | shape: (n, 1)
        x_test is a pandas dataframe where the instances belong to the labels y_test.
        y_prob is the probability of class 1 in the intervall [0,1].
        y_pred is the predicted class, 0 or 1.
        """
        
        ce = lambda y_test, y_prob:CalibrationEvaluator(y_test, y_prob, outsample=True)
        
        self.measure_functions["brier_score"] = lambda                x_test, y_prob ,y_pred, y_test:  ce(y_test, y_prob).brier
        self.measure_functions["spiegelhalter_z_statistic"] = lambda  x_test, y_prob ,y_pred, y_test:  ce(y_test, y_prob).z_test().statistic
        self.measure_functions["log_loss"] = lambda                   x_test, y_prob ,y_pred, y_test:  skm.log_loss(y_test, y_prob, normalize=True)

        self.measure_functions["auc_roc"] = lambda    x_test, y_prob ,y_pred, y_test:   ce(y_test, y_prob).auroc
        
        self.measure_functions["accuracy"] = lambda   x_test, y_prob ,y_pred, y_test:   skm.accuracy_score(y_test, y_pred, normalize=True) 
        self.measure_functions["f1_score"] = lambda   x_test, y_prob ,y_pred, y_test:   skm.f1_score(y_test, y_pred, pos_label=1, average='binary')
        self.measure_functions["precision"] = lambda  x_test, y_prob ,y_pred, y_test:   skm.precision_score(y_test, y_pred, pos_label=1, average='binary')
        self.measure_functions["recall"] = lambda     x_test, y_prob ,y_pred, y_test:   skm.recall_score(y_test, y_pred,pos_label=1, average='binary')

        def calfram_measures(measure_name, y_test, y_prob, y_pred):
            #Convert shape: (n, 1) to shape: (n, 2)
            y_prob = np.hstack([1 - y_prob, y_prob])

            cf = CalibrationFramework()

            classes_scores = CalibrationFramework().select_probability(y_test # shape: (n, 1)
                                                                    ,y_prob # shape: (n, c), where c is the number of classes 
                                                                    ,y_pred # shape: (n, 1)
            )
        
            measures, _ = cf.calibrationdiagnosis(classes_scores, adaptive=True #automatic monotonic sweep method
            )
            measure = None
            match measure_name:
                case "eci_global":
                    measure = measures["class_1"]["ec_g"] # ECI_global
                case "eci_balance":
                    measure = measures["class_1"]["ec_dir"] # ECI_balance
                case "ece_freq":
                     measure = measures["class_1"]["ece_fp"] #ECE_frequency 
            return measure
        
        self.measure_functions["eci_global"] = lambda     x_test, y_prob ,y_pred, y_test: calfram_measures("eci_global", y_test, y_prob, y_pred)
        self.measure_functions["eci_balance"] = lambda    x_test, y_prob ,y_pred, y_test: calfram_measures("eci_balance", y_test, y_prob, y_pred)
        self.measure_functions["ece_freq"] = lambda       x_test, y_prob ,y_pred, y_test: calfram_measures("ece_freq", y_test, y_prob, y_pred)



class AnalyzePerformance:
    def __init__(self, study_ver ,results):
        pass

    def init_v1(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def save_to_disk(self,file_name):
        raise NotImplementedError
