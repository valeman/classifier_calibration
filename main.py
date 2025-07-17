from components.log_config import configure_logger, log_progress_snapshot 
from components.performance import PerformanceMeasures
from components.architectures import ArchitectureSuite
from components.eval_strat import EvaluationStrategy
from components.data import DatasetSuite
import os, traceback, time, random, torch
import components.utils as util
import numpy as np


SEED = 123456789
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1") #Unsafe. Don't do this in prod.
ds_suite_name = "Tabarena-v0.1-binary"
eval_strat_name = "5-fold-CV"
study_version = "v.1"
output_dir = "results"

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    lg, progress = configure_logger()

    lg.info("Defining experiment")
    datasets = DatasetSuite(ds_suite_name, random_seed=SEED)
    p_measures = PerformanceMeasures(study_version)
    architectures = ArchitectureSuite(study_version, random_seed=SEED)
    datasets_metadata = {}
    results = {}

    lg.info("Starting experiment")
    with progress:
        outer = progress.add_task("", total=datasets.n_datasets)

        for ds_idx, ds in enumerate(datasets, start=1):
            ds_desc = f"Dataset {ds_idx}/{datasets.n_datasets}: {ds.df_name}"
            progress.update(outer, description=f"[bold blue]{ds_desc}", advance=1)
            lg.info(f"{ds_desc}")  
            
            lg.info(f"Start common pre-processing")
            ds.convert_to_pandas()
            ds.df = ds.df.sample(1500) #TODO:REMOVE
            ds.pre_process("detect_categorical") #Tag object columns as categorical
            ds.pre_process("convert_nan_to_unique_val") #Replace nan in cat columns with a new uniqe value
            ds.pre_process("encode_categoricals") #To {0,1} if binary else {1,2,3,...}
            ds.pre_process("clean_numerical") #Ensure non-cat object columns only contain numbers.
            ds.pre_process("convert_nan_to_-1") #Fill nan in numeric features with -1
            lg.info(f"End common pre-processing")
            datasets_metadata[ds.df_name] = ds.meta_data

            inner = progress.add_task("", total=architectures.n_architectures)
            
            for arch_idx,arch in enumerate(architectures, start=1):
                arch_desc = f"Architecture {arch_idx}/{architectures.n_architectures}: {arch.name}"
                progress.update(inner,description=f"[green]{arch_desc}",advance=1)
                log_progress_snapshot(progress)

                if arch.name not in results.keys():
                    results[arch.name] = {}
                
                lg.info(f"Start evaluation")
                start_eval = time.perf_counter()
                eval_strat = EvaluationStrategy(eval_strat_name, ds, p_measures, random_seed=SEED)  
                try:
                    results[arch.name][ds.df_name] = eval_strat.run(arch)

                except Exception as err:
                    lg.exception(f"Failed to evaluate {arch.name} on {ds.df_name}")
                    results[arch.name][ds.df_name] = [{
                        "status": "failed"
                        ,"error_message": str(err)
                        ,"trace": traceback.format_exc()
                    }]
                end_eval = time.perf_counter()
                lg.info(f"End evaluation of {arch.name}. Wall time spent: {util.format_time(end_eval - start_eval)}")
            
            task = progress.tasks[inner]
            lg.info(f"All evaluations on {ds.df_name} ended. Wall time spent: {util.format_time(task.elapsed)}")
            progress.remove_task(inner) 
               
            if ds.df_name == "APSFailure":
                break #TODO: REMOVE

    progress.remove_task(outer)   
    lg.info("Experiment completed")
    
    lg.info("Export results")
    util.save_dict_to_disk(results, output_dir, "results.txt")
    util.save_dict_to_disk(datasets_metadata, output_dir, "dataset_md.txt")
    
    