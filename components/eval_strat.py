from sklearn.model_selection import StratifiedKFold
from typing import Generator
from components.data import Dataset 
from components.architectures import Architecture
from components.performance import PerformanceMeasures
import components.utils as util
import pandas as pd
import numpy as np
import tracemalloc
import logging
import psutil
import time
import gc

lg = logging.getLogger(__name__)

class EvaluationStrategy:
    """
    The EvaluationStrategy standardizes the evaluation of an Architecture using a defined evaluation strategy and performance measurements. 
    """
    def __init__(self, strategy:str, ds:Dataset, p_measures:PerformanceMeasures, random_seed:int=123):
        self.strategy = strategy
        self.ds = ds
        self.p_measures = p_measures
        self.random_seed = random_seed
        self.gen_sets = None
        match strategy:
            case "5-fold-CV":
                self.k_fold_CV(ds, k=5)
            case _:
                raise NotImplementedError
            
    def run(self, arch:Architecture) -> list[dict]:
        results = []
        process = psutil.Process()
        count = 0
        for x_train, y_train, x_test, y_test in self.gen_sets():
            lg.info(f"Start run {count}")
            run_start = time.perf_counter()
            gc.collect()
            tracemalloc.start()
            cpu_fit_start = process.cpu_times()
            start_fit = time.perf_counter()

            arch.fit(self.ds.meta_data, x_train, y_train)
            
            end_fit = time.perf_counter()
            cpu_fit_end = process.cpu_times()
            _, peak_ram_fit = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            gc.collect()
            tracemalloc.start()
            cpu_pre_start = process.cpu_times()
            start_pre = time.perf_counter()

            y_prob = arch.predict_prob(x_test)

            end_pre = time.perf_counter()
            cpu_pre_end = process.cpu_times()
            _, peak_ram_pre = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        
            y_pred = arch.predict(y_prob=y_prob)

            perf_measures = self.p_measures.calc_perf(x_test, y_prob ,y_pred,np.asarray(y_test))
            perf_measures = self.log_computational_performance(perf_measures
                                                               , start_fit
                                                               , end_fit
                                                               , start_pre
                                                               , end_pre
                                                               , cpu_fit_start
                                                               , cpu_fit_end
                                                               , cpu_pre_start
                                                               , cpu_pre_end
                                                               , peak_ram_fit
                                                               , peak_ram_pre
            )

            
            
            perf_measures["status"]= "success"
            results.append(perf_measures)
            run_end = time.perf_counter()
            lg.info(f"End run {count}. Wall time spent: {util.format_time(run_end - run_start)}")
            count += 1
        return results
    
    def k_fold_CV(self, ds:Dataset, k=5)-> Generator[tuple[pd.DataFrame],None,None]:
        """
        Implements k-fold cross validation.  
        Outer heldout fold is test set, remaining is the training set. 
        
        Each fold is made through randomized stratified sampling without replacement.
        The strata is the target, ensuring that each class is represented in each fold according
        to it's relative empirical frequency.

        Args:
            ds (Dataset): A tabular dataset
        
        Yields:
            Generator[tuple[pd.DataFrame]]: A generator yielding the sets: x_train, y_train, x_test, y_test
        """
        # Extract target name and full DataFrame
        target_col = ds.meta_data['target']
        df = ds.df
        
        # Split features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        def gen_sets():
            # Stratified split
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_seed)
            for train_idx, test_idx in cv.split(X, y):
                X_train = X.iloc[train_idx].reset_index(drop=True)
                y_train = y.iloc[train_idx].reset_index(drop=True)
                X_test = X.iloc[test_idx].reset_index(drop=True)
                y_test = y.iloc[test_idx].reset_index(drop=True)
                yield X_train, y_train, X_test, y_test

        self.gen_sets = gen_sets
    
    def log_computational_performance(self, perf_measures:dict
                                      , start_fit
                                      , end_fit
                                      , start_pre
                                      , end_pre
                                      , cpu_fit_start
                                      , cpu_fit_end
                                      , cpu_pre_start
                                      , cpu_pre_end
                                      , peak_ram_fit
                                      , peak_ram_pre
                                    ) -> dict:
        perf_measures["wall_time_fit_sec"] = end_fit - start_fit
        perf_measures["wall_time_predict_sec"] = end_pre - start_pre
        perf_measures['cpu_time_user_fit_sec'] = cpu_fit_end.user - cpu_fit_start.user
        perf_measures['cpu_time_system_fit_sec'] = cpu_fit_end.system - cpu_fit_start.system
        perf_measures['cpu_time_user_predict_sec'] = cpu_pre_end.user - cpu_pre_start.user
        perf_measures['cpu_time_system_predict_sec'] = cpu_pre_end.system - cpu_pre_start.system
        perf_measures["peak_ram_fit_mb"] = peak_ram_fit  / 1e6
        perf_measures["peak_ram_predict_mb"] = peak_ram_pre / 1e6
        return perf_measures            
