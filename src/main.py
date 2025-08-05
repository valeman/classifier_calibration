from torch import manual_seed, set_num_threads, set_num_interop_threads
from components.log_config import configure_logger, log_progress_snapshot 
from components.performance import PerformanceMeasures
from components.architectures import ArchitectureSuite
from components.eval_strat import EvaluationStrategy
from components.data import DatasetSuite
from multiprocessing import cpu_count
import components.utils as util
import os, traceback, time
from random import seed
from numpy import random

SEED = 123456789
ds_suite_name = "Tabarena-v0.1-binary"
eval_strat_name = "5-fold-CV"
study_version = "v.1"
output_dir = "results"

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1") #Unsafe. Don't do this in prod.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = str(0)
os.environ['PYTHONHASHSEED'] = str(SEED)
seed(SEED)
random.seed(SEED)
manual_seed(SEED) 

n_cores = str(max(1, cpu_count() - 2)) 
os.environ["OMP_NUM_THREADS"] = n_cores
os.environ["OPENBLAS_NUM_THREADS"] = n_cores
os.environ["MKL_NUM_THREADS"] = n_cores
os.environ["NUMEXPR_NUM_THREADS"] = n_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = n_cores  
os.environ["KMP_AFFINITY"] = "noverbose"
set_num_interop_threads(int(n_cores))
set_num_threads(int(n_cores))


if __name__ == "__main__":
    lg, progress = configure_logger()

    lg.info("Defining experiment")
    datasets = DatasetSuite(ds_suite_name, random_seed=SEED)
    p_measures = PerformanceMeasures(study_version)
    architectures = ArchitectureSuite(study_version, random_seed=SEED, n_cores=n_cores)
    datasets_metadata = {}
    experiment_metadata = {"machine_md":{"n_cores":int(n_cores), "max_ram_mib":util.get_max_ram_mib()}, "random_seed":SEED}
    results = {}

    lg.info(f"Starting experiment: {util.time_now()}")
    start_exp = time.perf_counter()
    with progress:
        outer = progress.add_task("", total=datasets.n_datasets)

        for ds_idx, ds in enumerate(datasets, start=1):
            start_ds = time.perf_counter()
            ds_desc = f"Dataset {ds_idx}/{datasets.n_datasets}: {ds.df_name}"
            progress.update(outer, description=f"[bold blue]{ds_desc}", advance=1)
            lg.info(f"{ds_desc}")  
            
            lg.info(f"Start common pre-processing")
            ds.convert_to_pandas()
            ds.pre_process("detect_categorical") #Tag object columns as categorical
            ds.pre_process("convert_nan_to_unique_val") #Replace nan in cat columns with a new uniqe value
            ds.pre_process("encode_categoricals") #To {0,1} if binary else {1,2,3,...}
            ds.pre_process("clean_numerical") #Ensure non-cat object columns only contain numbers.
            ds.pre_process("convert_nan_to_0") #Fill nan in numeric features with 0
            lg.info(f"End common pre-processing")
            datasets_metadata[str(ds.df_name)] = ds.meta_data
            
            inner = progress.add_task("", total=architectures.n_architectures)
            for arch_idx,arch in enumerate(architectures, start=1):
                arch_desc = f"Architecture {arch_idx}/{architectures.n_architectures}: {arch.name}"
                progress.update(inner,description=f"[green]{arch_desc}",advance=1)
                log_progress_snapshot(progress)

                if arch.name not in results.keys():
                    results[str(arch.name)] = {}
                
                lg.info(f"Start evaluation")
                start_eval = time.perf_counter()
                eval_strat = EvaluationStrategy(eval_strat_name, ds, p_measures, random_seed=SEED)  
                try:
                    results[str(arch.name)][str(ds.df_name)] = eval_strat.run(arch)

                except Exception as err:
                    lg.exception(f"Failed to evaluate {arch.name} on {ds.df_name}")
                    results[str(arch.name)][str(ds.df_name)] = [{
                        "status": "failed"
                        ,"error_message": str(err)
                        ,"trace": traceback.format_exc()
                    }]
                end_eval = time.perf_counter()
                lg.info(f"End evaluation of {arch.name}. Wall time spent: {util.format_time(end_eval - start_eval)}")
            
            end_ds = time.perf_counter()
            lg.info(f"All evaluations on {ds.df_name} ended. Wall time spent: {util.format_time(end_ds - start_ds)}")
            progress.remove_task(inner) 

    end_exp = time.perf_counter()
    lg.info(f"Experiment completed. Wall time spent {util.format_time(end_exp-start_exp)}")

    lg.info("Export results")
    util.save_dict_to_disk(results, output_dir, "results.txt")
    util.save_dict_to_disk(datasets_metadata, output_dir, "datasets_md.txt")
    util.save_dict_to_disk(experiment_metadata, output_dir, "experiment_md.txt")
    
    