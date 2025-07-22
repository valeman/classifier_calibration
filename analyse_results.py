from matplotlib.patches import Patch
from itertools import chain
from typing import Tuple, Generator
import matplotlib.pyplot as plt
import matplotlib
import components.utils as util
import numpy as np


matplotlib.use('Agg')

def merge_results():
    raise NotImplementedError

def merge_mds():
    raise NotImplementedError

def sanity_checks() -> dict:
    """Return a dictonary mapping performance measure names to functions.
        The functions return a string when the value of the measure is in an
        unexpected set. 

    Returns:
        dict
    """
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
    """Quality control the res and md dict to ensure everything is there and is as expected.

    Args:
        res (dict): The results dict output from main.py
        md (dict): The meta_data dict output from main.py
    """
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


def invert_res_by_ds(res:dict) -> Tuple[dict,int]:
    """
    Takes the res dict and returns an inverted version
    Also returns the number of runs per ds and arch. 
    
    res = {arch:{ds:[{measure:value},{measure:value},...]}}
    inv_res = {ds:{arch:[{measure:value},{measure:value},...]}}
    
    Args:
        res (dict)

    Returns:
        Tuple[dict,int]
    """
    inv_res = {}
    n_runs = 0
    for arch_name, ds_dict in res.items():    
        for ds_name, runs in ds_dict.items():
            if ds_name not in inv_res.keys():
                    inv_res[ds_name] = {}

            inv_res[ds_name][arch_name] = runs
            n_runs = max(n_runs, len(runs))
    return inv_res, n_runs


def rank_across_runs_by_ds_me(inv_res:dict, ranking_m:dict, n_runs:int, archs:list, n_archs:int) -> dict:
    """
    For each dataset and measure, rank each arch per run.
    Aggregate across runs to get probability vectors per dataset, measure and arch.
    The probability vectors reflect the likelihood of the arch having a given ranking.
    If you randomly choose a run on the given dataset ranking by the measure.

    Args:
        inv_res (dict)
        ranking_m (dict): Key is the measure name, value is a bool idicating whether to rank by desc order. 
        n_runs (int)
        archs (list)
        n_archs (int)

    Returns:
        ranking_dict (dict): {ds:{measure:{arch:np.array}}}
            The innermost array specifies the probability per rank by index.
                Index 0 is 1st, index 1 is 2nd etc..
    """
    ranking_dict = {}
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
    return ranking_dict


def rank_across_runs_by_me(agg_key:str, ranking_m:dict, ranking_dict:dict, archs:list, n_archs:int) -> dict:
    """
    For each measure, rank each arch per run for each dataset.
    Aggregate across dataset to get probability vectors per measure and arch.
    The probability vectors reflect the likelihood of the arch having a given ranking.
    If you randomly choose a run and dataset ranking by the measure.

    Args:
        agg_key (str): The key to store the results under.
        ranking_m (dict): See: def rank_across_runs_by_ds
        ranking_dict (dict): See: def rank_across_runs_by_ds
        archs (list): See: def rank_across_runs_by_ds
        n_archs (int): See: def rank_across_runs_by_ds

    Returns:
        dict: See: def rank_across_runs_by_ds
            It's ranking_dict with a new outer key added {agg_key}
    """
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
    return ranking_dict


def rank_across_runs(agg_key:str, ranking_dict:dict, archs:list, n_archs:int) -> dict:
    """
    Rank each arch per run for each dataset and measure.
    Aggregate across mesures to get probability vectors per arch.
    The probability vectors reflect the likelihood of the arch having a given ranking.
    If you randomly choose a run,dataset and measure.

    Args:
        agg_key (str): _description_
        ranking_dict (dict): _description_
        archs (list): _description_
        n_archs (int): _description_

    Returns:
        dict: _description_
    """
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
    return ranking_dict


def calc_expected_ranking(ranking_dict:dict) -> dict:
    """Transforms all the probability vectors into scalars by calculating expectation. 

    Args:
        ranking_dict (dict): See: def rank_across_runs

    Returns:
        dict: Same as ranking_dict just scalars instead of vectors. 
    """
    exp_ranking_dict = {}
    for ds_name, ds_dict in ranking_dict.items():
        exp_ranking_dict[ds_name] = {}
        for ms_name, ms_dict in ds_dict.items():
            exp_ranking_dict[ds_name][ms_name] = {}
            for arch_name, prob_vec in ms_dict.items():
                ranks = np.arange(1, len(prob_vec) + 1)    
                exp_rank = np.dot(ranks, prob_vec)
                exp_ranking_dict[ds_name][ms_name][arch_name] = exp_rank
    return exp_ranking_dict


def calc_marginals(inv_res:dict, archs:list) -> dict:
    """
    Takes the results from inv_res and returns a new dict on the same structure. 
    But now all measure values are marginals indicating how measures changed,
    after applying a post-hoc calibraiton method. 
    Measure values: After - Before

    Args:
        inv_res (dict): See: def invert_res_by_ds
        archs (list)

    Returns:
        dict
    """
    marg_inv_res = {k:{} for k,v in inv_res.items()}
    base = [a for a in archs if "." not in a] 
    pp = [a for a in archs if "." in a]
    for ds_name, ds_dict in inv_res.items():
        for base_a in base:
            pp_as = [a for a in pp if base_a in a]
            
            for pp_a in pp_as:
                marg_inv_res[ds_name][pp_a] = [
                    {k:p[k]-b[k] for k in b if k != "status"} for b,p in zip(ds_dict[base_a], ds_dict[pp_a])
                    ]
    return marg_inv_res

def transform_per_grouping(marg_inv_res:dict, archs:list) -> Tuple[str,int,int,list,dict]:
    #Get all the unique models and post-hoc calibration methods
    models = set([a.split(".")[0] for a in archs])
    models = sorted(list(models))
    phcm = set([a.split(".")[1] for a in archs if "." in a])
    phcm =  sorted(list(phcm)) 
    ds_key = list(marg_inv_res.keys())[0]
    
    #Yield results aggregated across models, grouped by post-hoc calibration method.
    agg_marg_inv_res = {
        ds: {
            a: list(
                chain.from_iterable(
                    ds_dict[f"{m}.{a}"] for m in models
                )
            )
            for a in phcm
        }
        for ds, ds_dict in marg_inv_res.items()
    }
    i_n_runs=len(agg_marg_inv_res[ds_key][phcm[0]])
    i_n_archs=len(phcm)
    yield "phc_method",i_n_runs, i_n_archs, phcm, agg_marg_inv_res
    
    #Yield results per model
    for model in models:
        grouping = model
        i_marg_inv_res = {ds:{arch:v for arch,v in ds_dict.items() if model in arch} for ds,ds_dict in marg_inv_res.items()}
        i_archs = sorted(list(i_marg_inv_res[ds_key].keys()))
        i_n_archs = len(i_archs)
        i_n_runs = len(i_marg_inv_res[ds_key][i_archs[0]])
        yield grouping, i_n_runs, i_n_archs, i_archs, i_marg_inv_res 
    

def analyse_results(res:dict, md:dict, output_dir:str, assets_dir:str) -> None:
    """
    Takes the performance measures in res and metadata in md to produce and export the below analysis:

    Ranking is performed by placing each arch for a dataset and run in a relative position (1st, 2nd, 3rd...) according to a performance measure.
    An arch's probability of being in a position observed fraction: (number of times in position)/(total runs). 
    You then get a distribution reflecting the probability of an arch having each of the positions.

    A single ranking across archs is made by sorting the archs by the highest expected position. 

	1. Rank absolute calibration performance rating by ECI global,log-loss,brier score and z-score of each architecture across all datasets and runs.
        Ranking is performed per run. 
        Ranking is also aggregated across measures. 
    
    2. Plot wall time train and predict of each architecture against n_cells in train/test across datasets
        #TODO:Also CPU time and peak RAM?

	3. Rank marginal calibration performance rating by ECI global,log-loss brier score and z-score of each post-hoc method across all datasets
		Ranking is performed per run.
        Ranking is also aggregated across measures. 

        First across models and then per model. 
        
    4. Display distribution of relative improvement in calibration performance by method. 
        First across models then per model

	5. Per post-hoc calibration method. 
        See delta in other performance measures.  
            Did any calibraiton methods degrade other metrics?
        First across models then per model

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
    #Create dir to store images 
    dir_path = output_dir + "/" + assets_dir
    dir_path = util.create_pwd_dir(dir_path)
    #Get the name of all architectures
    archs = sorted(list(res.keys()))
    n_archs = len(archs)

    #Invert res by dataset 
    inv_res, n_runs = invert_res_by_ds(res)
    
    ranking_m = {
    "brier_score":False,
    "log_loss":False,
    "eci_global":True,
    #"abs_clip_spiegelhalter_z_statistic":False,
    } 
    comp_cost_m = ["wall_time_fit_sec","wall_time_predict_sec"] 
    agg_key = "aggregate"    
    
    #1:
    #Probability of an arch having a rank by a measure if you randomly choose a run for a dataset
    ranking_dict = rank_across_runs_by_ds_me(inv_res, ranking_m, n_runs, archs, n_archs)
    #Probability of an arch having a rank by a measure if you randomly choose a run and dataset
    ranking_dict = rank_across_runs_by_me(agg_key, ranking_m, ranking_dict, archs, n_archs)
    #Probability of an arch having a rank  if you randomly choose a measure, run and dataset
    ranking_dict = rank_across_runs(agg_key, ranking_dict, archs, n_archs)    
    #Calculate expected rankings.
    exp_ranking_dict = calc_expected_ranking(ranking_dict)

    ##Export bar plot of the aggregated expected rankings. 
    i_dir_path = util.create_pwd_dir(dir_path + "/abs/")
    for ms_name in exp_ranking_dict[agg_key].keys():
        plot_rankings(exp_ranking_dict[agg_key][ms_name]
                      ,i_dir_path + f"exp_agg_{ms_name}.png"
                      ,title=f"Expected performance rank across runs and datasets"
                      ,x_label=f"Performance measure: {ms_name}")

    #2:
    for cc_m in comp_cost_m: 
        outfile = f'{dir_path}/{cc_m}.png'
        plot_line_cc(inv_res, md, archs, cc_m, outfile)
        
    #3: 
    #Transform performance measures into marginals
    marg_inv_res = calc_marginals(inv_res, archs)

    #Perform ranking on marginals per post-hoc calibration method across models and per model.
    for grouping, i_n_runs, i_n_archs, i_archs, i_marg_inv_res in transform_per_grouping(marg_inv_res, archs):
        #Probability of an arch having a rank by a measure if you randomly choose a run for a dataset
        i_marg_ranking_dict = rank_across_runs_by_ds_me(i_marg_inv_res, ranking_m, i_n_runs, i_archs, i_n_archs)
        #Probability of an arch having a rank by a measure if you randomly choose a run and dataset
        i_marg_ranking_dict = rank_across_runs_by_me(agg_key, ranking_m, i_marg_ranking_dict, i_archs, i_n_archs)
        #Probability of an arch having a rank  if you randomly choose a measure, run and dataset
        i_marg_ranking_dict = rank_across_runs(agg_key, i_marg_ranking_dict, i_archs, i_n_archs)    
        #Calculate expected rankings.
        i_marg_exp_ranking_dict = calc_expected_ranking(i_marg_ranking_dict)

        ##Export bar plot of the aggregated expected rankings. 
        for ms_name in i_marg_exp_ranking_dict[agg_key].keys():
            i_dir_path = util.create_pwd_dir(dir_path + f"/marg/{grouping}/")
            plot_rankings(i_marg_exp_ranking_dict[agg_key][ms_name]
                        ,i_dir_path + f"exp_agg_{ms_name}.png"
                        ,title=f"Expected performance rank across runs and datasets"
                        ,x_label=f"Performance measure: marginal {ms_name}"
                        ,top_n=1
                        ,bottom_n=1
                        )

    #4: 
    #Transform performance measures into relative change (pct)
    
    #5: 
    #Use relative change data 
    #Examine distribution of relative change in non-calibration measures. 
    

def plot_line_cc(inv_res:dict, md:dict, archs:list, cc_m:str, outfile:str) -> None:
    """
    #TODO: FILL
    """
     # Adjust figure size dynamically based on number of legends
    base_width = 10
    extra_width = 0.5 * len(archs)  # scale with number of legends
    fig_width = 10
    fig_height = 6
    plt.figure(figsize=(fig_width, fig_height))

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of cells (rows × features)")
    plt.ylabel(cc_m.replace("_", " ").title() + " (s)")
    plt.title(cc_m.replace("_", " ").title() + " vs. Problem Size")

    for arch in archs:
        x_vals = []
        y_vals = []
        for ds_name, ds_dict in inv_res.items():
            if "fit" in cc_m:
                n_inst = round(md[ds_name]["n_rows"] * 4/5)
            else:
                n_inst = round(md[ds_name]["n_rows"] * 1/5)
            n_features = md[ds_name]["n_columns"] - 1
            n_cells = n_inst * n_features

            x_run = []
            y_run = []
            for run in ds_dict[arch]:
                x_run.append(n_cells)
                y_run.append(run[cc_m])
            x_vals.append(np.mean(x_run))
            y_vals.append(np.mean(y_run))

        plt.plot(x_vals, y_vals, marker='o', label=arch)
       # Move legend to the right of the plot
    plt.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make room on right for legend

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_rankings(data: dict, outfile: str, title:str, x_label:str, top_n: int = 5, bottom_n: int = 5) -> None:
    """
    Export a horizontal bar chart of model rankings and highlight top and bottom performers.

    Args:
        data (dict): Mapping from model names to ranking metric (lower is better).
        outfile (str): File path to save the figure
        title (str)
        x_label (str)
        top_n (int, optional): Number of top performers to highlight in orange. Defaults to 5.
        bottom_n (int, optional): Number of bottom performers to highlight in green. Defaults to 5.
    """
    # Prepare and sort data
    names = list(data.keys())
    values = np.array([data[n] for n in names], dtype=float)
    order = np.argsort(values)
    sorted_names = [names[i] for i in order]
    sorted_values = values[order]

    # Colors: default, top, bottom
    default_color = '#1f77b4'
    top_color = '#ff7f0e'
    bottom_color = '#2ca02c'
    colors = [default_color] * len(sorted_values)
    for i in range(min(top_n, len(colors))):
        colors[i] = top_color
    for i in range(1, min(bottom_n, len(colors)) + 1):
        colors[-i] = bottom_color

    # Dynamic figure size
    n = len(sorted_names)
    height = max(6, 0.3 * n)
    fig, ax = plt.subplots(figsize=(10, height))

    # Plot bars
    bars = ax.barh(sorted_names, sorted_values, color=colors)
    ax.invert_yaxis()  # best at top

    # Labels and title
    ax.set_xlabel(x_label)
    ax.set_title(title)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (0.01 * sorted_values.max()),
            bar.get_y() + bar.get_height() / 2,
            f'{width:.2f}',
            va='center', fontsize=8
        )

    # Legend
    legend_elements = [
        Patch(facecolor=top_color, label=f'Top {top_n}'),
        Patch(facecolor=default_color, label='Middle performers'),
        Patch(facecolor=bottom_color, label=f'Bottom {bottom_n}')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Layout adjustments
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ranking_hist(data:dict, arch:str) -> None:
    """
    Plots a horizontal bar chart of ranking probabilities for a given architecture.
    
    Args:
        data (dict): dict mapping architecture names to 1D numpy arrays of probabilities 
        architecture (str): str key of the architecture's name
    """
    probs = data[arch]
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
    ax.set_title(f"Ranking Distribution for {arch}")
    plt.tight_layout()
    plt.savefig('ranking_lr.png', dpi=150, bbox_inches='tight')



if __name__ == "__main__":
    
    output_dir = "results_log"
    assets_dir = "assets"
    #Load all the data
    res = util.load_dict(output_dir, "results.txt")
    md = util.load_dict(output_dir, "datasets_md.txt")
    
    #Merge data from different processes
    #res = merge_results(res)
    #md = merge_mds(md)

    #Make sure all the data is there and makes sense
    qc_input(res, md)

    #Analyse the data and export results 
    analyse_results(res, md, output_dir, assets_dir)
    

    