import components.config as cf 
import components.data as data
import components.architectures as archs
import components.performance as perf
from typing import Generator
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


SEED = cf.SEED

class EvaluationStrategy:
    def __init__(self, strategy:str, ds:data.Dataset, arch:archs.Architecture,  p_measures:perf.PerformanceMeasures):
        self.strategy = strategy
        self.ds = ds
        self.arch = arch 
        self.p_measures = p_measures
        
        self.gen_sets = None
        match strategy:
            case "v.1":
                self.init_v1(ds, arch)

    def run(self) -> list[dict]:
        results = []
        for x_train, y_train, x_cal, y_cal, x_test, y_test in self.gen_sets():
            pass
            self.arch.fit(self.ds.meta_data, x_train, y_train, x_cal, y_cal)
            y_prob= self.arch.predict_prob(x_test)
            y_pred = self.arch.predict(y_prob=y_prob)

            perf_measures = self.p_measures.calc_perf(x_test, y_prob ,y_pred,np.asarray(y_test))
            results.append(perf_measures)
            
        return results
    
    def init_v1(self, ds:data.Dataset, arch:archs.Architecture)-> Generator[tuple[pd.DataFrame],None,None]:
        """
        Nested 5-fold cross validation, with 5 outer folds and 5 inner folds.  
        Outer heldout fold is test set, inner heldout fold is calibration set. 
        No inner cross validation if no calibration set is requested by the architecture. 

        Each fold is made through randomized stratified sampling without replacement.
        The strata is the target, ensuring that each class is represented in each fold according
        to it's relative empirical frequency.

        Args:
            ds (data.Dataset): A tabular dataset
            arch (archs.Architecture): The architecture

        Yields:
            Generator[tuple[pd.DataFrame]]: A generator yielding the sets: x_train, y_train, x_calibration, y_calibration, x_test, y_test
        """
        # Extract target name and full DataFrame
        target_col = ds.meta_data['target']
        df = ds.df
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        
        def gen_sets():
            # Outer stratified split
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for train_idx, test_idx in outer_cv.split(X, y):
                X_train = X.iloc[train_idx].reset_index(drop=True)
                y_train = y.iloc[train_idx].reset_index(drop=True)
                X_test = X.iloc[test_idx].reset_index(drop=True)
                y_test = y.iloc[test_idx].reset_index(drop=True)

                # If calibration set requested, perform inner stratified split
                if arch.calibration_set:
                    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
                    for inner_train_idx, cal_idx in inner_cv.split(X_train, y_train):
                        X_in_train = X_train.iloc[inner_train_idx].reset_index(drop=True)
                        y_in_train = y_train.iloc[inner_train_idx].reset_index(drop=True)
                        X_cal = X_train.iloc[cal_idx].reset_index(drop=True)
                        y_cal = y_train.iloc[cal_idx].reset_index(drop=True)
                        yield X_in_train, y_in_train, X_cal, y_cal, X_test, y_test
                else:
                    # No calibration set: inner split skipped
                    yield X_train, y_train, None, None, X_test, y_test

        self.gen_sets = gen_sets
     