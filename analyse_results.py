import os
import json

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
        ,"recall":between_0_1
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
            if isinstance(runs,dict): #TODO:REMOVE
                runs = [runs]
            for i, run in enumerate(runs):
             
                status = run.get("status", "failed")
                if status == "failed":
                    err_msg = run.get("error_message", "<no message>")
                    print(f"{arch} failed during run {i} on {ds}: {err_msg}")
                    continue
                
                run.pop("total_ram_architecture_mb") #TODO:REMOVE

                for measure, value in run.items():
                    if measure in ["status", "error_message", "trace"]:
                        continue
                    try:
                        float(value)
                    except ValueError:
                        print(f"ValueError: Run {i} ({arch} on {ds}): cannot convert {measure!r}={value!r} to float")      
                    
                    eval = scs[measure](value)
                    if eval:
                        print(f"ValueError: Run {i} ({arch} on {ds}): {measure!r}={value!r}  Value {eval}")      
                

def analyse_results(res:dict, md:dict) -> dict:
    raise NotImplementedError


def export_analysis(ana, output_dir):
    raise NotImplementedError


if __name__ == "__main__":
    res = load_dict(output_dir, "results_v1.txt")
    md = load_dict(output_dir, "datasets_md_v1.txt")
    
    qc_input(res, md)

    #ana = analyse_results(res, md)
    #export_analysis(ana, output_dir)