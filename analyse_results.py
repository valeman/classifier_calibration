import os
import json
import numpy as np
import components.utils as util


output_dir = "results"

def load_dict(output_dir: str, file_name: str) -> dict:
    """
    Load a dictionary from a JSON dumped .txt file.

    Parameters
    ----------
    output_dir : str
        Subdirectory (under cwd) where the file lives.
    file_name : str
        Name of the .txt file (e.g. "mydict.txt").

    Returns
    -------
    Dict
        The dictionary that was saved.

    Raises
    ------
    FileNotFoundError
        If the target file does not exist.
    ValueError
        If the file's contents aren't a JSON object.
    json.JSONDecodeError
        If the file isn't valid JSON.
    """
    path = os.path.join(os.getcwd(), output_dir, file_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find file at {path!r}")

    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object (dict) in {path!r}, got {type(obj)}")
    return obj


def sanity_checks() -> dict:
        anything = lambda value: None
        between_m1_p1 = lambda value: None if -1 <= value <= 1 else "not in [-1, 1]"
        between_0_1 = lambda value: None if 0 <= value <= 1 else "not in [0, 1]"
        greater_then_0 = lambda value: None if 0 <= value else "not in [0, inf]"
        
        scs = {
        "spiegelhalter_z_statistic":anything
        ,"eci_balance":between_m1_p1            
        ,"eci_global":between_0_1
        ,"accuracy":between_0_1
        ,"f1_score":between_0_1
        ,"precision":between_0_1
        ,"recall_1":between_0_1
        ,"recall_0":between_0_1
        ,"brier_score":greater_then_0
        ,"log_loss":greater_then_0
        ,"auc_roc":greater_then_0
        ,"ece_freq":greater_then_0
        ,"wall_time_fit_sec":greater_then_0
        ,"wall_time_predict_sec":greater_then_0
        ,"cpu_time_user_fit_sec":greater_then_0
        ,"cpu_time_system_fit_sec":greater_then_0
        ,"cpu_time_user_predict_sec":greater_then_0
        ,"cpu_time_system_predict_sec":greater_then_0
        ,"peak_ram_fit_mb":greater_then_0
        ,"peak_ram_predict_mb":greater_then_0
        }
        return scs


def qc_input(res:dict, md:dict) -> None:
    scs = sanity_checks()
    for arch, dss in res.items():
        for ds, runs in dss.items():
            for i, run in enumerate(runs):
             
                status = run.get("status", "failed")
                if status == "failed":
                    err_msg = run.get("error_message", "<no message>")
                    print(f"{arch} failed during run {i} on {ds}: {err_msg}")
                    continue

                for measure, value in run.items():
                    if measure in ["status", "error_message", "trace"]:
                        continue
                    if not util.all_numbers_and_finite(np.asarray([value])):
                        print(f"ValueError: Run {i} ({arch} on {ds}): {measure!r}={value!r} is not a finite number")      
                    
                    eval = scs[measure](value)
                    if eval:
                        print(f"ValueError: Run {i} ({arch} on {ds}): {measure!r}={value!r}  Value {eval}")      
                

def analyse_results(res:dict, md:dict) -> dict:
    """
    I have dataset metadata (md), and performance measures per architecture, dataset and run (res).

    1. Average out measures across runs. Gives Expectation in measure value on OOS instances, given dataset and architecture.
        sum cpu time

	2. Rank absolute calibration performance rating by ECI global,log-loss,brier cal aggregate and comp cost  of each architecture across all datasets 
        Ranking pre-averaging. 
        Display average comp cost of each arch scaled by n_train, n_test across datasets.
        Highlight average comp cost across runs for a small, medium and large dataset per arch. 

	3. Rank marginal calibration performance rating by  ECI global,log-loss and brier  of each post-hoc method across all datasets
			Did any degrade calibration?

            post-hoc method rating grouped by model across datasets marginal
					Which is best with respect to the model across datasets

			post-hoc method rating across datasets and models marginal
					which is best across models and datasets
							
	4. See if post-hoc calibration degrades overall performance. 
        Did any calibraiton methods degrade other metrics?
    
    Dump averages per dataset to appendix    
    """  
    raise NotImplementedError


def export_analysis(ana, output_dir):
    raise NotImplementedError


if __name__ == "__main__":
    res = load_dict(output_dir, "results.txt")
    md = load_dict(output_dir, "datasets_md.txt")
    
    qc_input(res, md)

    #ana = analyse_results(res, md)
    #export_analysis(ana, output_dir)

    