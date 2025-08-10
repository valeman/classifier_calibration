from modules.analysis.ranking import (rank_abs_arch,
                                      rank_del_phcms, 
                                      get_learners, 
                                      get_phcm
)
from modules.analysis.plots import (plot_scatter_cc,
                                    plot_box_cu,
                                    plot_rankings,
                                    plot_changes,
                                    plot_dual_rankings
)
import modules.common.utils as util
from itertools import combinations
from typing import Tuple
import math, re
import numpy as np

output_dir = "archive/results_2" #TODO:AMEND
assets_dir = "assets"

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
    ,"wall_time_fit_sec": greater_then_excl_0
    ,"cpu_time_total_fit_sec": greater_then_excl_0
    ,"cpu_time_user_fit_sec": greater_then_excl_0
    ,"cpu_time_system_fit_sec": greater_then_0
    ,"peak_ram_fit_mib": greater_then_excl_0
    ,"peak_swap_fit_mib": greater_then_0
    ,"peak_zswap_fit_mib": greater_then_0
    ,"io_read_total_fit_mib": greater_then_0
    ,"io_write_total_fit_mib": greater_then_0
    ,"wall_time_pre_sec": greater_then_excl_0
    ,"cpu_time_total_pre_sec": greater_then_excl_0
    ,"cpu_time_user_pre_sec": greater_then_excl_0
    ,"cpu_time_system_pre_sec": greater_then_0
    ,"peak_ram_pre_mib": greater_then_0
    ,"peak_swap_pre_mib": greater_then_0
    ,"peak_zswap_pre_mib": greater_then_0
    ,"io_read_total_pre_mib": greater_then_0
    ,"io_write_total_pre_mib": greater_then_0
    ,"peak_ram_predict_mb":greater_then_0
    }
    return scs


def qc_input(res:dict, ds_md:dict, exp_md:dict) -> None:
    """Quality control the res,ds_md and exp_md dict to ensure everything is there and is as expected.

    Args:
        res (dict): The results dict output from main.py
        ds_md (dict): The datasets meta_data dict output from main.py
        exp_md (dict): The experiment meta_data dict output from main.py
    """
    #Check that all architectures are in the res dict
    if len(res.keys()) != 96:
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


def enrich_res(res:dict, ds_md:dict, exp_md:dict):
    """
    Enriches res by deriving new performance measures from the results and meta data.
    """
    n_cores = exp_md["machine_md"]["n_cores"]
    max_ram = exp_md["machine_md"]["max_ram_mib"]

    for arch_name, ds_dict in res.items():    
        for ds_name, runs in ds_dict.items():
            for run in runs:
                for stage in ["fit", "pre"]:
                    max_cpu_time_total_sec = run[f"wall_time_{stage}_sec"] * n_cores 
                    run[f"cpu_time_total_{stage}_util_pct"] = (run[f"cpu_time_total_{stage}_sec"]/max_cpu_time_total_sec) * 100
                    run[f"peak_ram_{stage}_util_pct"] = (run[f"peak_ram_{stage}_mib"]/max_ram) *100
    return res

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

def calc_marginals(inv_res:dict, archs:list) -> dict:
    """
    Takes the results from inv_res and returns a new dict of the same structure. 
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


def is_effectively_zero(x, tol=1e-12):
    """
    Return True if x (int or float) is zero within absolute tolerance tol.
    """
    return math.isclose(x, 0.0, abs_tol=tol)


def rel_change(before, marginal):
    if is_effectively_zero(marginal):
        return 0 
    if is_effectively_zero(before):
        return "ZeroDivisionError"
    return np.sign(marginal) * np.abs(marginal/before) * 100


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

def rank_archs(inv_res:dict, ranking_m:dict, agg_key:str, dir_path:str, n_runs:int, archs:list, n_archs:int, post_fix:str="/abs/"):
    _, exp_ranking_dict = rank_abs_arch(inv_res, ranking_m, agg_key, n_runs, archs, n_archs)

    ##Export bar plot of the aggregated expected rankings. 
    i_dir_path = util.create_pwd_dir(dir_path + post_fix)
    for ms_name in exp_ranking_dict[agg_key].keys():
        plot_rankings(exp_ranking_dict[agg_key][ms_name]
                      ,i_dir_path + f"exp_rank_{ms_name}.png"
                      ,title=f"Expected performance rank across folds and datasets"
                      ,x_label=f"Performance measure: {ms_name}"
                      ,top_n=3 if "lrns" in post_fix else 5
                       ,bottom_n=3 if "lrns" in post_fix else 5
                      )
    
    i_dir_path = util.create_pwd_dir(dir_path + post_fix + "/dual/")
    for meas_1, meas_2 in list(combinations(exp_ranking_dict[agg_key].keys(), 2)):
        plot_dual_rankings(exp_ranking_dict[agg_key]
                           , archs
                           , i_dir_path + f"exp_rank_{meas_1}_{meas_2}.png"
                           , meas_1
                           , meas_2
                           )


def rank_phcms(delta_inv_res:dict, ranking_m:dict, agg_key:str, dir_path:str, n_runs:int, archs:list, post_fix:str="/delta/"):
    _, exp_ranking_dict = rank_del_phcms(delta_inv_res, ranking_m, agg_key, n_runs, archs)
    
    ##Export bar plot of the aggregated expected rankings. 
    for ms_name in exp_ranking_dict[agg_key].keys():
        i_dir_path = util.create_pwd_dir(dir_path + post_fix + f"/{ms_name}/")
        for lrn in exp_ranking_dict[agg_key][ms_name]:
            plot_rankings(exp_ranking_dict[agg_key][ms_name][lrn]
                        ,i_dir_path + f"exp_rank_{lrn}_{ms_name}.png"
                        ,title=f"Expected performance rank across folds and datasets"
                        ,x_label=f"Performance measure: Δ {ms_name}"
                        ,top_n=1
                        ,bottom_n=1
            )

    m_list = [k for k in exp_ranking_dict[agg_key].keys() if k != agg_key]
    m_vals = {k:v[agg_key] for k,v in exp_ranking_dict[agg_key].items() if k != agg_key}
    i_dir_path = util.create_pwd_dir(dir_path + post_fix + "/dual/")
    for meas_1, meas_2 in list(combinations(m_list, 2)):
        plot_dual_rankings(  m_vals
                           , get_phcm(archs)
                           , i_dir_path + f"exp_rank_{meas_1}_{meas_2}.png"
                           , meas_1
                           , meas_2
                           , monkey_fix = True
                           )



def analyse_results(res:dict, ds_md:dict, exp_md:dict, output_dir:str, assets_dir:str) -> None:
    """
    Args:
        res (dict): A nested dictionary. 
            Key is the architecture. Value is a dict. For the inner dict:
                Key is the dataset
                Value is a list. For the inner list:
                    Each element is a dict. For the inner dict:
                        Key is the measure name. Value is the measure's numeric value.
        ds_md (dict): A nested dictionary.
            Key is the dataset. Value is a dict with meta data. 
        exp_md (dict): A nested dictionary.
            Contains seeds, machine specs etc. 
    """
    ranking_m = {
    "brier_score":False, #False: Ascending | Less is better
    "log_loss":False,
    "eci_global":True, #True: Descending | More is better
    "abs_clip_spiegelhalter_z_statistic":False,
    } 
    comp_cost_m = ["wall_time_fit_sec"
                   ,"cpu_time_total_fit_sec"
                   ,"peak_ram_fit_mib"
                   ,"wall_time_pre_sec"
                   ,"cpu_time_total_pre_sec"
                   ,"peak_ram_pre_mib"
                   ]
    comp_cost_m = {m:False for m in comp_cost_m}
    comp_util_m = ["cpu_time_total_fit_util_pct"
                   ,"peak_ram_fit_util_pct"
                   ,"cpu_time_total_pre_util_pct"
                   ,"peak_ram_pre_util_pct"
                   ]
    rel_delta_m = comp_cost_m | {"auc_roc":True}
    marg_delta_m = {
    "accuracy":True,    
    "recall_1":True,
    "recall_0":True,
    } 
    agg_key = "aggregate"    
    #Create dir to store images 
    dir_path = output_dir + "/" + assets_dir
    dir_path = util.create_pwd_dir(dir_path)
    
    #Prep data
    res = enrich_res(res, ds_md, exp_md)
    archs = sorted(list(res.keys()))
    n_archs = len(archs)
    inv_res, n_runs = invert_res_by_ds(res)
    lrns = get_learners(archs)
    n_lrns = len(lrns)
    res_lrns = {k:v for k,v in res.items() if k in lrns}
    inv_res_lrns, _ = invert_res_by_ds(res_lrns)
    marg_inv_res = calc_marginals(inv_res, archs)
    rel_inv_res = calc_relative(inv_res, marg_inv_res, archs)

    #1:
    rank_archs(inv_res, ranking_m | comp_cost_m, agg_key, dir_path, n_runs, archs, n_archs, post_fix="/abs/rank/archs/")
    rank_archs(inv_res_lrns, ranking_m | comp_cost_m, agg_key, dir_path, n_runs, lrns, n_lrns, post_fix="/abs/rank/lrns/")
    
    #2:
    c_dir_path = util.create_pwd_dir(dir_path + "/cost/")
    for cc_m in comp_cost_m: 
        outfile = f'{c_dir_path}/{cc_m}.png'
        plot_scatter_cc(inv_res, ds_md, archs, cc_m, outfile)
    
    cu_dir_path = util.create_pwd_dir(dir_path + "/cost/util/")
    for cu_m in comp_util_m:
        plot_box_cu(res, archs, cu_m, cu_dir_path)

    #3: 
    rank_phcms(marg_inv_res, ranking_m | comp_cost_m, agg_key, dir_path, n_runs, archs, post_fix="/delta/marg/rank/")
    
    #4:  
    #Export distribution of relative change in calibration measures per post-hoc calibration method across learners and per learner.
    plot_changes(rel_inv_res, ranking_m, archs, dir_path, "relative")
    
    #5: 
    #Export distribution of relative change in non-calibration measures. 
    plot_changes(rel_inv_res, rel_delta_m, archs, dir_path, "relative")
    #Export distribution of marginal change in non-calibration measures. 
    plot_changes(marg_inv_res, marg_delta_m, archs, dir_path, "marginal")
    

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
    #Load files
    res = util.load_dict(output_dir, "results.txt")
    ds_md = util.load_dict(output_dir, "datasets_md.txt")
    exp_md = util.load_dict(output_dir, "experiment_md.txt")
    
    #Load and aggregate files from different sources
    #res = merge_dicts(output_dir, "results.txt")
    #ds_md = merge_dicts(output_dir, "datasets_md.txt")
    
    #res = {k:v for k,v in res.items() if not ("ttra" in k)} #TODO:REMOVE
    
    #Make sure all the data is there and makes sense
    qc_input(res, ds_md, exp_md)
    
    #Analyse the data and export results 
    analyse_results(res, ds_md, exp_md, output_dir, assets_dir)
    

    