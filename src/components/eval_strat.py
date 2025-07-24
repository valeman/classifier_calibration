from sklearn.model_selection import StratifiedKFold
from typing import Generator
from components.data import Dataset 
from components.architectures import Architecture
from components.performance import PerformanceMeasures
from components.resource_tracker import ResourceTracker
import components.utils as util
import pandas as pd
import numpy as np
import logging
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
        count = 0
        for x_train, y_train, x_test, y_test in self.gen_sets():
            gc.collect()
            lg.info(f"Start run {count}")
            run_start = time.perf_counter()
            
            start_fit = time.perf_counter()
            with ResourceTracker(sample_interval=0.05) as rt:
                arch.fit(self.ds.meta_data, x_train, y_train)
            end_fit = time.perf_counter()
            time_fit = end_fit - start_fit
            rsc_fit = rt.to_dict()
            
            gc.collect()
            start_pre = time.perf_counter()
            with ResourceTracker(sample_interval=0.05) as rt:
                y_prob = arch.predict_prob(x_test)
            end_pre = time.perf_counter()
            time_pre = end_pre - start_pre
            rsc_pre = rt.to_dict()

            y_pred = arch.predict(y_prob=y_prob)

            perf_measures = self.p_measures.calc_perf(x_test, y_prob ,y_pred,np.asarray(y_test))
            perf_measures = self.log_computational_performance("fit", perf_measures, time_fit, rsc_fit)
            perf_measures = self.log_computational_performance("pre", perf_measures, time_pre, rsc_pre)
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
    

    def log_computational_performance(self
                                      , section:str
                                      , perf_measures:dict
                                      , timed:float
                                      , rsc:dict
                                    ) -> dict:
        perf_measures[f"wall_time_{section}_sec"] = timed
        perf_measures[f'cpu_time_total_{section}_sec'] = rsc["total_cpu_time_sec"]
        perf_measures[f'cpu_time_user_{section}_sec'] = rsc["user_cpu_time_sec"]
        perf_measures[f'cpu_time_system_{section}_sec'] = rsc["system_cpu_time_sec"]
        perf_measures[f"peak_ram_{section}_mib"] = rsc["peak_memory_mib"] 
        perf_measures[f"peak_swap_{section}_mib"] = rsc["peak_swap_mib"] 
        perf_measures[f"peak_zswap_{section}_mib"] = rsc["peak_zswap_mib"] 
        perf_measures[f"io_read_total_{section}_mib"] = rsc["total_io_read_mib"] 
        perf_measures[f"io_write_total_{section}_mib"] = rsc["total_io_write_mib"] 
        return perf_measures            
