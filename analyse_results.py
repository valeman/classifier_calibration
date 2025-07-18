import components.utils as util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, json

output_dir = "results_log"

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
        between_m1_p1 = lambda value: None if -1 <= value <= 1 else "not in [-1, 1]"
        between_0_1 = lambda value: None if 0 <= value <= 1 else "not in [0, 1]"
        greater_then_0 = lambda value: None if 0 <= value else "not in [0, inf]"
        
        scs = {
        "abs_clip_spiegelhalter_z_statistic":greater_then_0
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
                    if measure == "status":
                        continue

                    if not util.all_numbers_and_finite(np.asarray([value])):
                        print(f"ValueError: Run {i} ({arch} on {ds}): {measure!r}={value!r} is not a finite number")      
                    
                    eval = scs[measure](value)
                    if eval:
                        print(f"ValueError: Run {i} ({arch} on {ds}): {measure!r}={value!r}  Value {eval}")      
                

def analyse_results(res:dict, md:dict) -> dict:
    """
    I have dataset metadata (md), and performance measures per architecture, dataset and run (res).

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
    
    1. Average out measures across runs. Gives Expectation in measure value on OOS instances, given dataset and architecture.
        sum cpu time
    Dump averages per dataset to appendix    
    """  

    """    
	2. Rank absolute calibration performance rating by ECI global,log-loss,brier cal aggregate and comp cost  
    of each architecture across all datasets 
        Ranking pre-averaging. 
        Display average comp cost of each arch scaled by n_train, n_test across datasets.
        Highlight average comp cost across runs for a small, medium and large dataset per arch. 
    """
    
    #Get the name of all architectures
    archs = sorted(list(res.keys()))
    n_archs = len(archs)

    #Invert res by dataset 
    inv_res = {}
    n_runs = 0
    for arch_name, ds_dict in res.items():    
        for ds_name, runs in ds_dict.items():
            if ds_name not in inv_res.keys():
                    inv_res[ds_name] = {}

            inv_res[ds_name][arch_name] = runs
            n_runs = max(n_runs, len(runs))
    
    ranking_m = {
    "brier_score":False,
    "log_loss":False,
    "eci_global":False,
    #"abs_clip_spiegelhalter_z_statistic":False,
    
    } 
    ranking_dict = {}
    #Rank each arch per dataset across runs. 
    #Gives Likelihood of each arch having a given rank by a measure for a given dataset across runs.
    #Probability of an arch having a rank by a measure if you randomly choose a run for a dataset
    for ds_name, arch_dict in inv_res.items():
        ranking_dict[ds_name] = {}
        for measure,desc in ranking_m.items():
            ranking_dict[ds_name][measure] = {a:np.zeros(n_archs) for a in archs}
                            
            for i in range(n_runs):            
                m_vals = [] # where to store the values
                for arch_name in archs:    
                    m_vals.append(
                        round(arch_dict[arch_name][i][measure],5)
                    )    
                m_vals = list(zip(archs,m_vals))
                ranking = [name for name, _ in sorted(m_vals, key=lambda x: x[1], reverse=desc)]
                
                for j,arch_name in enumerate(ranking):
                    c_rank = np.zeros(n_archs)
                    c_rank[j] = 1
                    p_rank = ranking_dict[ds_name][measure][arch_name]
                    if np.all(p_rank == 0):
                        ranking_dict[ds_name][measure][arch_name] = c_rank
                    else:
                        ranking_dict[ds_name][measure][arch_name] = (p_rank + c_rank)/2


    #Aggregate ranking across all datasets
    #Gives Likelihood of each arch having a given rank by a measure across runs and datasets.
    #Probability of an arch having a rank by a measure if you randomly choose a run and dataset
    agg_key = "aggregate"
    ranking_dict[agg_key] = {k:{a:np.zeros(n_archs) for a in archs} for k,_ in ranking_m.items()}
    for ds_name, ms_dict in ranking_dict.items():    
        
        if ds_name == agg_key:
            continue

        for measure, m_dict in ms_dict.items():
            for arch_name, ds_rank in m_dict.items():
                p_rank = ranking_dict[agg_key][measure][arch_name] 
                if np.all(p_rank == 0):
                    ranking_dict[agg_key][measure][arch_name] = ds_rank
                else:
                    ranking_dict[agg_key][measure][arch_name] = (p_rank + ds_rank)/2
                
    #Aggregate ranking across measures
    #Gives Likelihood of each arch having a given rank across measure, across runs and across datasets.
    #Probability of an arch having a rank  if you randomly choose a measure, run and dataset
    ranking_dict[agg_key][agg_key] = {a:np.zeros(n_archs) for a in archs}
    for m_name, m_dict in ranking_dict[agg_key].items():
        
        if m_name == agg_key:
            continue
        
        for arch_name, m_rank in m_dict.items():
            p_rank = ranking_dict[agg_key][agg_key][arch_name]
            
            if np.all(p_rank == 0):
                ranking_dict[agg_key][agg_key][arch_name] = m_rank
            else:
                ranking_dict[agg_key][agg_key][arch_name] = (p_rank + m_rank)/2
    
    ranking_dict[agg_key][agg_key]["lr"]
    plot_ranking_hist(ranking_dict[agg_key][agg_key], 'lr')


    #Pick arch with highest probability per ranking position. 
    #Arch with highest expected ranking positions
    raise NotImplementedError


def plot_ranking_hist(data, architecture):
    """
    Plots a horizontal bar chart of ranking probabilities for a given architecture.
    
    Parameters:
    - data: dict mapping architecture names to 1D numpy arrays of probabilities.
    - architecture: str key in data to visualize.
    """
    probs = data[architecture]
    ranks = np.arange(1, len(probs) + 1)  # 1st place = 1, 2nd place = 2, ...

    fig, ax = plt.subplots()
    ax.barh(ranks, probs)
    
    # Invert y-axis so rank 1 appears at the top
    ax.invert_yaxis()
    
    # Label ticks
    ax.set_yticks(ranks)
    ax.set_yticklabels([f"{i}st" if i == 1 else f"{i}nd" if i == 2 else f"{i}rd" if i == 3 else f"{i}th" for i in ranks])
    
    ax.set_xlabel("Probability")
    ax.set_ylabel("Rank Position")
    ax.set_title(f"Ranking Distribution for {architecture}")
    plt.tight_layout()
    plt.savefig('ranking_lr.png', dpi=150, bbox_inches='tight')



def export_analysis(ana, output_dir):
    raise NotImplementedError


if __name__ == "__main__":
    res = load_dict(output_dir, "results.txt")
    md = load_dict(output_dir, "datasets_md.txt")
    
    qc_input(res, md)

    ana = analyse_results(res, md)
    #export_analysis(ana, output_dir)

    