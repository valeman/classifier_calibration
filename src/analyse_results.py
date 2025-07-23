from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from itertools import chain
from typing import Tuple
import components.utils as util
import matplotlib.pyplot as plt
import matplotlib
import math, re
import numpy as np

matplotlib.use('Agg')


def sanity_checks() -> dict:
    """
    Return a dictonary mapping performance measure names to functions.
    The functions return a string when the value of the measure is in an
    unexpected set. 

    Returns:
        dict
    """
    between_m1_p1 = lambda value: None if -1 <= value <= 1 else "not in [-1, 1]"
    between_0_1 = lambda value: None if 0 <= value <= 1 else "not in [0, 1]"
    greater_then_0 = lambda value: None if 0 <= value else "not in [0, inf]"
    greater_then_excl_0 = lambda value: None if 0 < value else "not in <0, inf]"
    
    scs = {
    "abs_clip_spiegelhalter_z_statistic":greater_then_0
    ,"eci_balance":between_m1_p1            
    ,"eci_global":between_0_1
    ,"accuracy":between_0_1
    ,"f1_score":between_0_1
    ,"precision":between_0_1
    ,"recall_1":between_0_1
    ,"recall_0":between_0_1
    ,"brier_score":greater_then_excl_0
    ,"log_loss":greater_then_excl_0
    ,"auc_roc":greater_then_excl_0
    ,"ece_freq":greater_then_0
    ,"wall_time_fit_sec":greater_then_excl_0
    ,"wall_time_predict_sec":greater_then_excl_0
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
        ranking_m (dict): See: def rank_across_runs_by_ds_me
        ranking_dict (dict): See: def rank_across_runs_by_ds_me
        archs (list): See: def rank_across_runs_by_ds_me
        n_archs (int): See: def rank_across_runs_by_ds_me

    Returns:
        dict: See: def rank_across_runs_by_ds_me
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
        agg_key (str): See: def rank_across_runs_by_me
        ranking_dict (dict): See: def rank_across_runs_by_me
        archs (list)
        n_archs (int)

    Returns:
        dict: See: def rank_across_runs_by_me
            It's ranking_dict with a new inner key added {agg_key}
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


def get_learners(archs:list) -> list:
    learners = set([a.split(".")[0] for a in archs])
    learners = sorted(list(learners))
    return learners


def get_phcm(archs:list) -> list:
    phcm = set([a.split(".")[1] for a in archs if "." in a])
    phcm =  sorted(list(phcm))
    return phcm
    

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
    marg_inv_res = {k:{} for k in inv_res}
    learners = get_learners(archs)
    phcm = get_phcm(archs)
    pp = [a for a in archs if any(p in a for p in phcm)]
    
    for ds_name, ds_dict in inv_res.items():
        for learner in learners:
            pp_as = [a for a in pp if learner in a]
            
            for pp_a in pp_as:
                marg_inv_res[ds_name][pp_a] = [
                    {k:p[k]-b[k] for k in b if k != "status"} for b,p in zip(ds_dict[learner], ds_dict[pp_a])
                    ]
    return marg_inv_res


def transform_per_grouping(marg_inv_res:dict, archs:list) -> Tuple[str,int,int,list,dict]:
    #Get all the unique learners and post-hoc calibration methods
    learners = get_learners(archs)
    phcm = get_phcm(archs)
    ds_key = list(marg_inv_res.keys())[0]
    
    #Yield results aggregated across learners, grouped by post-hoc calibration method.
    agg_marg_inv_res = {
        ds: {
            a: list(
                chain.from_iterable(
                    ds_dict[f"{m}.{a}"] for m in learners
                )
            )
            for a in phcm
        }
        for ds, ds_dict in marg_inv_res.items()
    }
    i_n_runs=len(agg_marg_inv_res[ds_key][phcm[0]])
    i_n_archs=len(phcm)
    yield "phc_method",i_n_runs, i_n_archs, phcm, agg_marg_inv_res
    
    #Yield results per learner
    for learner in learners:
        grouping = learner
        i_marg_inv_res = {ds:{arch:v for arch,v in ds_dict.items() if learner in arch} for ds,ds_dict in marg_inv_res.items()}
        i_archs = sorted(list(i_marg_inv_res[ds_key].keys()))
        i_n_archs = len(i_archs)
        i_n_runs = len(i_marg_inv_res[ds_key][i_archs[0]])
        yield grouping, i_n_runs, i_n_archs, i_archs, i_marg_inv_res 


def is_effectively_zero(x, tol=1e-12):
    """
    Return True if x (int or float) is zero within absolute tolerance tol.
    """
    return math.isclose(x, 0.0, abs_tol=tol)


def rel_change(before, marginal):
    if is_effectively_zero(marginal):
        return 0 
    if is_effectively_zero(before):
        return "Zero divison"
    return (marginal/before) * 100


def calc_relative(inv_res:dict, marg_inv_res:dict, archs:list) -> dict:
    """
    Calculates relative change in perf metrics per run.

    Args:
        inv_res (dict): See: def invert_res_by_ds
        marg_inv_res (dict): See: def calc_marginals
        archs (list)

    Returns:
        dict: Same structure as marg_inv_res only with relative change in pct (50 not 0.5 for 50%)
        instead of marginals
    """
    rel_inv_res = {k:{} for k in inv_res}
    learners = get_learners(archs)
    phcm = get_phcm(archs)
    pp = [a for a in archs if any(p in a for p in phcm)]
    
    
    for ds_name, abs_ds_dict in inv_res.items():
        marg_ds_dict = marg_inv_res[ds_name]
        for learner in learners:
            pp_as = [a for a in pp if learner in a]
            for pp_a in pp_as:
                
                rel_inv_res[ds_name][pp_a] = [
                    {k:rel_change(b[k], mp[k]) for k in b if k != "status"} 
                    for b,mp in zip(abs_ds_dict[learner], marg_ds_dict[pp_a])
                    ]
     
    return rel_inv_res


def analyse_results(res:dict, md:dict, output_dir:str, assets_dir:str) -> None:
    """
    Takes the performance measures in res and metadata in md to produce and export the below analysis:

    Note: Ranking is performed by placing each arch for a dataset and run in a relative position (1st, 2nd, 3rd...) according to a performance measure.
    An arch's probability of being in a position is the empirical frequency: (number of runs in a position)/(total runs). (runs/folds)
    You then get a distribution reflecting the probability of an arch having each of the positions.

    A single ranking across archs is made by sorting the archs by the highest expected position. 

	1. Rank absolute calibration performance rating by ECI global,log-loss,brier score and z-score of each architecture across all datasets and runs.
        Ranking is also aggregated across measures. 
    
    2. Plot wall time train/predict, CPU time train/predict, peak RAM train/predict and of each architecture against n_cells in train/test across datasets.
        Only done for the base learners, relative change in these measures come later.

	3. Rank marginal calibration performance rating by ECI global,log-loss brier score and z-score of each post-hoc method across all datasets
        Ranking is also aggregated across measures. 
        First across learners and then per learner. 
        
    4. Display distribution of relative change in calibration performance by method. 
        First across learners then per learner
        Plot also includes expected value to the plot.

	5. Display distribution of change in non-calibration performance by method. 
        First across learners then per learner
            Relative change:
                wall time train/predict, CPU time train/predict, peak RAM train/predict
                AUC_ROC
                
            Marginal change
                Accuracy
                Recall_1, Recall_2
         

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
    "brier_score":False, #False: Ascending | Less is better
    "log_loss":False,
    "eci_global":True, #False: Descending | More is better
    "abs_clip_spiegelhalter_z_statistic":False,
    } 
    comp_cost_m = ["wall_time_fit_sec"
                   #,"cpu_time_user_fit_sec"
                   #,"cpu_time_system_fit_sec"
                   #,"peak_ram_fit_mb"
                   ,"wall_time_predict_sec"
                   #,"cpu_time_user_predict_sec"
                   #,"cpu_time_system_predict_sec"
                   #,"peak_ram_predict_mb"
                   ]
    rel_delta_m = {m:False for m in comp_cost_m}
    rel_delta_m["auc_roc"] = True
    marg_delta_m = {
    "accuracy":True,    
    "recall_1":True,
    "recall_0":True,
    } 
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
    c_dir_path = util.create_pwd_dir(dir_path + "/cost/")
    for cc_m in comp_cost_m: 
        outfile = f'{c_dir_path}/{cc_m}.png'
        plot_scatter_cc(inv_res, md, archs, cc_m, outfile)
        
    #3: 
    #Transform performance measures into marginals
    marg_inv_res = calc_marginals(inv_res, archs)

    #Perform ranking on marginals per post-hoc calibration method across learners and per learner.
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
    rel_inv_res = calc_relative(inv_res, marg_inv_res, archs)
    #Export distribution of relative change in calibration measures per post-hoc calibration method across learners and per learner.
    for grouping, _, _, _, i_rel_inv_res in transform_per_grouping(rel_inv_res, archs):
        i_dir_path = util.create_pwd_dir(dir_path + f"/rel/{grouping}/")    
        plot_changes(i_rel_inv_res, ranking_m, i_dir_path, grouping, change="rel")

    #5: 
    #Use relative change data 
    #Examine distribution of relative change in non-calibration measures. 
    for grouping, _, _, _, i_rel_inv_res in transform_per_grouping(rel_inv_res, archs):
        i_dir_path = util.create_pwd_dir(dir_path + f"/rel/{grouping}/")    
        plot_changes(i_rel_inv_res, rel_delta_m, i_dir_path, grouping, change="rel")
    
    #Use marginal change data 
    #Examine distribution of marginal change in non-calibration measures. 
    for grouping, _, _, _, i_marg_inv_res in transform_per_grouping(marg_inv_res, archs):
        i_dir_path = util.create_pwd_dir(dir_path + f"/marg/{grouping}/")    
        plot_changes(i_marg_inv_res, marg_delta_m, i_dir_path, grouping, change="marg")
   

def plot_scatter_cc(inv_res:dict, md:dict, archs:list, cc_m:str, outfile:str) -> None:
    """
    #TODO: FILL
    """
    archs = get_learners(archs)
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
                n_inst = round(md[ds_name]["n_rows"] * 4/5) #TODO: Pass down from main
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

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        # Plot transparent base points
        scatter = plt.scatter(x_vals, y_vals, alpha=0.3, label=None)

        # Overlay less transparent points in same color
        plt.scatter(x_vals, y_vals, alpha=0.3, color=scatter.get_facecolor()[0], label=None)

        # Fit polynomial in log-log space
        log_x = np.log10(x_vals)
        log_y = np.log10(y_vals)
        degree = min(2, len(log_x) - 1)  # degree 2 unless very few points
        coeffs = np.polyfit(log_x, log_y, deg=degree)
        poly = np.poly1d(coeffs)
        x_smooth = np.logspace(np.log10(min(x_vals)), np.log10(max(x_vals)), 200)
        y_smooth = 10 ** poly(np.log10(x_smooth))

        # Plot polynomial curve
        plt.plot(x_smooth, y_smooth, color=scatter.get_facecolor()[0],alpha=1, linewidth=2, label=arch)

    # Move legend to the right of the plot
    plt.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Make room on right for legend

    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()



def plot_rankings(data: dict, outfile: str, title:str, x_label:str, top_n: int = 5, bottom_n: int = 5) -> None:
    """
    Export a horizontal bar chart of learner rankings and highlight top and bottom performers.

    Args:
        data (dict): Mapping from learner names to ranking metric (lower is better).
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


def plot_changes(nested_res: dict,
                          meas_pref: dict,
                          output_dir: str,
                          grouping:str,
                          change:str,
                          grid_points: int = 300,
                          tail_std_mult: float = 4.0) -> None:
    """
    For each measurement in meas_pref, collect all change values per architecture
    from nested_res and plot kernel-density distributions stacked vertically per architecture,
    highlighting "good" vs "bad" regions and annotating area left/right of zero.

    Ensures the density grid extends beyond raw min/max by a multiple of std so the integrated
    area under the curve approximates 1.

    Produces one PNG file per measurement in output_dir, with one subplot per architecture.

    Args:
        nested_res (dict): dict of dataset -> dict of arch -> list of {measurement: relative_change}
        meas_pref (dict): dict of measurement -> bool (True if more is better)
        output_dir (str): path where PNGs will be saved
        grouping (str)
        change (str): Whether the change is relative or marginal. 
        grid_points (int, optional): number of points to evaluate KDE on. Defaults to 300.
        tail_std_mult (float, optional): number of std deviations to extend beyond data mean. Defaults to 4.0.
    """
    assert change in ["rel", "marg"]   

    for meas, more_is_better in meas_pref.items():
        arch_vals = {}
        for ds_dict in nested_res.values():
            for arch, runs in ds_dict.items():
                for run in runs:
                    if meas in run:
                        arch_vals.setdefault(arch, []).append(run[meas])
        if not arch_vals:
            continue

        # Combine all values for global stats
        all_vals = np.concatenate(list(arch_vals.values()))
        mean = np.mean(all_vals)
        std = np.std(all_vals)
        # Determine extended range to capture KDE tails
        vmin = mean - tail_std_mult * std
        vmax = mean + tail_std_mult * std
        # Fallback to raw data if std=0
        if std == 0 or vmin >= all_vals.min():
            vmin = all_vals.min() - 1e-6
        if std == 0 or vmax <= all_vals.max():
            vmax = all_vals.max() + 1e-6

        x_grid = np.linspace(vmin, vmax, grid_points)

        # Create stacked subplots
        n_arch = len(arch_vals)
        fig, axes = plt.subplots(n_arch, 1,
                                 sharex=True,
                                 figsize=(10, 2.5 * n_arch),
                                 constrained_layout=True)
        if n_arch == 1:
            axes = [axes]

        for ax, (arch, vals) in zip(axes, arch_vals.items()):
            kde = gaussian_kde(vals)
            y = kde(x_grid)

            # Masks for good/bad
            if more_is_better:
                bad_region = (-np.inf, 0)
                good_region = (0, np.inf)
            else:
                good_region = (-np.inf, 0)
                bad_region = (0, np.inf)

            good_mask = np.logical_and(x_grid >= good_region[0] if np.isfinite(good_region[0]) else True,
                                       x_grid <= good_region[1] if np.isfinite(good_region[1]) else True)
            bad_mask  = ~good_mask

            #Expected value
            expected_value = np.trapz(x_grid * y, x_grid)   

            # Areas
            area_good = kde.integrate_box_1d(*good_region)
            area_bad  = kde.integrate_box_1d(*bad_region)
       
            # Plot
            ax.fill_between(x_grid[good_mask], y[good_mask], alpha=0.6, label='good')
            ax.fill_between(x_grid[bad_mask], y[bad_mask], alpha=0.6, color='red', label='bad')
            ax.plot(x_grid, y, linewidth=1.2, color='black')
            ax.axvline(0, color='black', linewidth=1)
            
            
            # Annotate
            ann = f"E[V]: {expected_value:.3f}"
            if change == "rel":
                ann += "%"    
            ann += f"\nArea good: {area_good:.3f}\nArea bad: {area_bad:.3f}"
            
            ax.text(0.99, 0.8,
                    ann,
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize='small',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

            ax.set_ylabel(arch)
            ax.tick_params(axis='y', left=False, labelleft=False)
        if change == "marg":
            axes[-1].set_xlabel('Marginal Change')
        elif change == "rel":
            axes[-1].set_xlabel('Relative Change (%)')
        fig.suptitle(f"KDE of change in '{meas}'. Grouped by {grouping}", fontsize=14)

        out_file = output_dir + f"{meas}.png"
        plt.savefig(out_file, dpi=300)
        plt.close(fig)


def sort_key(item):
    # Extract all numbers from the string
    nums = list(map(int, re.findall(r'\d+', item)))
    # Pad with 0 if there's only one number
    if len(nums) == 1:
        nums.append(0)
    return tuple(nums)  # Sort by (first_num, second_num)


def merge_dicts(output_dir, file_name):
    output_dir = util.create_pwd_dir(output_dir)    
    sub_dirs = util.get_subdirs(output_dir)
    sub_dirs = sorted(sub_dirs, key=sort_key)
    merged = {}
    for dirr in sub_dirs:
        try:
            path = f"{output_dir}/{dirr}"
            merged.update(util.load_dict(path, file_name))
        except FileNotFoundError:
            print(f"No {file_name} in {path}")
    return merged


if __name__ == "__main__":
    output_dir = "archive"
    assets_dir = "assets"
    
    #Load all the data
    res = util.load_dict(output_dir, "results.txt")
    md = util.load_dict(output_dir, "datasets_md.txt")
    
    #Merge data from different sources
    #res = merge_dicts(output_dir, "results.txt")
    #md = merge_dicts(output_dir, "datasets_md.txt")
    
    #Make sure all the data is there and makes sense
    qc_input(res, md)

    #Analyse the data and export results 
    analyse_results(res, md, output_dir, assets_dir)
    

    