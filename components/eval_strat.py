import time
import tracemalloc
import psutil
import gc
import os
import components.config as cf 
import components.data as data
import components.architectures as archs
import components.performance as perf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from typing import Generator

SEED = cf.SEED

def get_mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6  # in MB


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
        process = psutil.Process()
        for x_train, y_train, x_cal, y_cal, x_test, y_test in self.gen_sets():
            
            gc.collect()
            ram_pre_arch = get_mem_mb() 
            tracemalloc.start()
            cpu_fit_start = process.cpu_times()
            start_fit = time.perf_counter()

            self.arch.fit(self.ds.meta_data, x_train, y_train, x_cal, y_cal)
            
            end_fit = time.perf_counter()
            cpu_fit_end = process.cpu_times()
            _, peak_ram_fit = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            ram_post_arch = get_mem_mb()
            
            gc.collect()
            tracemalloc.start()
            cpu_pre_start = process.cpu_times()
            start_pre = time.perf_counter()

            y_prob= self.arch.predict_prob(x_test)

            end_pre = time.perf_counter()
            cpu_pre_end = process.cpu_times()
            _, peak_ram_pre = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        
            y_pred = self.arch.predict(y_prob=y_prob)

            perf_measures = self.p_measures.calc_perf(x_test, y_prob ,y_pred,np.asarray(y_test))

            perf_measures["wall_time_fit_sec"] = end_fit - start_fit
            perf_measures["wall_time_predict_sec"] = end_pre - start_pre
            perf_measures['cpu_time_user_fit_sec'] = cpu_fit_end.user - cpu_fit_start.user
            perf_measures['cpu_time_system_fit_sec'] = cpu_fit_end.system - cpu_fit_start.system
            perf_measures['cpu_time_user_predict_sec'] = cpu_pre_end.user - cpu_pre_start.user
            perf_measures['cpu_time_system_predict_sec'] = cpu_pre_end.system - cpu_pre_start.system
      
            perf_measures["peak_ram_fit_mb"] = peak_ram_fit  / 1e6
            perf_measures["peak_ram_predict_mb"] = peak_ram_pre / 1e6
            perf_measures["total_ram_architecture_mb"] = ram_post_arch - ram_pre_arch
            
            perf_measures["n_train"] = len(x_train)
            perf_measures["n_cal"] = 0 if x_cal is None else len(x_cal)
            perf_measures["n_test"]= len(x_test)

            perf_measures["status"]= "success"
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
     