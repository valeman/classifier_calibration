import components.utils as util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def merge_results():
    raise NotImplementedError

def merge_mds():
    raise NotImplementedError

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
    #Check that all architectures are in the res dict
    if len(res.keys()) != 60:
        print("Architectures are missing")
    #Check that all architectures were ran on all datasets in the res dict
    for arch_name in res.keys():
        if len(res[arch_name].keys()) != 30:
            print(f"Datasets missing for {arch_name}") 

    #Check that all measure values make sense for all runs for all archs on all datasets
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
    Takes the performance measures in res and metadata in md to produce and export the below analysis:

    Ranking is performed by placing each arch for a dataset and run in a relative position (1st, 2nd, 3rd...) according to a performance measure.
    A archs probability of being in a position is incremented each time it empirically has that position. 
    You then get a distribution reflecting the probability of an arch having each of the positions. 
    A single ranking across archs is made by sorting the archs by the highest expected position. 

	1. Rank absolute calibration performance rating by ECI global,log-loss,brier score and z-score of each architecture across all datasets.
        Ranking is performed per run. 
        Ranking is also aggregated across measures. 
    
    2. Rank absolute computational cost of each architecture by wall time train and predict across all datasets
        Ranking is performed per run.

    3. Display average wall time train and predict cost of each architecture scaled by n_train and n_test across datasets
        
	4. Rank marginal calibration performance rating by ECI global,log-loss brier score and z-score of each post-hoc method across all datasets
		First per model and then across models. 
        Ranking is performed per run.
        Ranking is also aggregated across measures. 

    5. Display average relative improvement in calibration performance by method. 

	6. Per post-hoc calibration method. See delta in other performance measures.  
        Did any calibraiton methods degrade other metrics?

    Args:
        res (dict): A nested dictionary. 
            Key is the architecture. Value is a dict. For the inner dict:
                Key is the dataset
                Value is a list. For the inner list:
                    Each element is a dict. For the inner dict:
                        Key is the measure name. Value is the measure's numeric value.
        md (dict): A nested dictionary.
            Key is the dataset. Value is a dict with meta data. 
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
    
    #1:
    ranking_m = {
    "brier_score":False,
    "log_loss":False,
    "eci_global":False,
    #"abs_clip_spiegelhalter_z_statistic":False,
    } 
    ranking_dict = {}
    
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
    
    #Calculate expected rankings.
    
    ranking_dict.keys()
    ranking_dict["aggregate"].keys()
    ranking_dict["aggregate"]["aggregate"].keys() 
    ranking_dict["aggregate"]["aggregate"]["cb"]
    #Rank each arch by expected rating. 


    #ranking_dict[agg_key][agg_key]["lr"]
    #plot_ranking_hist(ranking_dict[agg_key][agg_key], 'lr')




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
    
    output_dir = "results_log"
    #Load all the data
    res = util.load_dict(output_dir, "results.txt")
    md = util.load_dict(output_dir, "datasets_md.txt")
    
    #Merge data from different processes
    #res = merge_results(res)
    #md = merge_mds(md)

    #Make sure all the data is there and makes sense
    qc_input(res, md)

    #Analyse the data
    ana = analyse_results(res, md)
    
    #Export the resulting analysis 
    #export_analysis(ana, output_dir)

    