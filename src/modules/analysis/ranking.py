import numpy as np


def get_learners(archs:list) -> list:
    learners = set([a.split(".")[0] for a in archs])
    learners = sorted(list(learners))
    return learners

def get_phcm(archs:list) -> list:
    phcm = set([a.split(".")[1] for a in archs if "." in a])
    phcm =  sorted(list(phcm))
    return phcm

def rank_abs_arch(inv_res:dict, ranking_m:dict, agg_key:str, n_runs:int, archs:list, n_archs:int):
    #Counts of an arch having a rank by a measure among all runs for each dataset
    ranking_dict = rank_across_runs_by_ds_me(inv_res, ranking_m, n_runs, archs, n_archs)
    #Counts of an arch having a rank by a measure among all runs and datasets
    ranking_dict = rank_across_runs_by_me(agg_key, ranking_m, ranking_dict, archs, n_archs)
    #Counts of an arch having a rank among all measures, runs and datasets.
    ranking_dict = rank_across_runs(agg_key, ranking_dict, archs, n_archs)    
    #Calculate expected rankings.
    exp_ranking_dict = calc_expected_ranking(ranking_dict)
    return ranking_dict, exp_ranking_dict 
    
def rank_del_phcms(delta_inv_res:dict, ranking_m:dict, agg_key:str, n_runs:int, archs:list):
    #Counts of an phcm having a rank by a measure among all runs for each dataset and learner
    ranking_dict = rank_across_runs_by_ds_me_lr(delta_inv_res, ranking_m, n_runs, archs)
    #Counts of an phcm having a rank by a measure among all runs for each learner
    ranking_dict = delta_rank_across_runs_by_me_lr(agg_key, ranking_m, ranking_dict, archs)
    #Counts of an phcm having a rank by a measure among all runs
    ranking_dict = delta_rank_across_runs_by_me(agg_key, ranking_m, ranking_dict, archs)
    #Counts of an arch having a rank among all measures and runs.
    ranking_dict = delta_rank_across_runs(agg_key, ranking_dict, archs)  
    #Calculate expected rankings.
    exp_ranking_dict = calc_expected_delta_ranking(ranking_dict)
    return ranking_dict, exp_ranking_dict

def rank_across_runs_by_ds_me(inv_res:dict, ranking_m:dict, n_runs:int, archs:list, n_archs:int) -> dict:
    """
    For each dataset and measure, rank each arch per run.
    Aggregate across runs to get ranking vectors per dataset, measure and arch.
    The ranking vectors reflect the counts an architecture had any of the ranks.

    Args:
        inv_res (dict)
        ranking_m (dict): Key is the measure name, value is a bool idicating whether to rank by desc order. 
        n_runs (int)
        archs (list)
        n_archs (int)

    Returns:
        ranking_dict (dict): {ds:{measure:{arch:np.array}}}
            The innermost array specifies the count per rank by index.
                Index 0 is 1st, index 1 is 2nd etc..
    """
    ranking_dict = {}
    for ds_name, arch_dict in inv_res.items():
        ranking_dict[ds_name] = {}
        for measure,desc in ranking_m.items():
            ranking_dict[ds_name][measure] = {a:np.zeros(n_archs) for a in archs}
                            
            for i in range(n_runs):            
                m_vals = [] 
                for arch_name in archs:    
                    m_vals.append(
                        round(arch_dict[arch_name][i][measure],5)
                    )    
                m_vals = list(zip(archs,m_vals))
                ranking = rank_list(m_vals, desc)            
                
                for name, rank in ranking:
                    rank_vec = ranking_dict[ds_name][measure][name]
                    rank_vec[rank-1]  += 1
                    ranking_dict[ds_name][measure][name] = rank_vec 

    return ranking_dict


def rank_across_runs_by_ds_me_lr(d_inv_res:dict, ranking_m:dict, n_runs:int, archs:list) -> dict:
    """
    For each dataset and measure, rank each phcm per run and model.
    Aggregate across runs to get ranking vectors per dataset, measure and model.
    The ranking vectors reflect the counts an phcm had any of the ranks.

    Args:
        d_inv_res (dict)
        ranking_m (dict): Key is the measure name, value is a bool idicating whether to rank by desc order. 
        n_runs (int)
        archs (list)
        n_archs (int)

    Returns:
        ranking_dict (dict): {ds:{measure:{lrn:{phcm:np.array}}}}
            The innermost array specifies the count per rank by index.
                Index 0 is 1st, index 1 is 2nd etc..
    """
    ranking_dict = {}
    lrns = get_learners(archs)
    phcms = get_phcm(archs)
    for ds_name, arch_dict in d_inv_res.items():
        ranking_dict[ds_name] = {}
        for measure,desc in ranking_m.items():
            ranking_dict[ds_name][measure] = {l:{p:np.zeros(len(phcms)) for p in phcms} for l in lrns}
            for lrn in lrns:            
                for i in range(n_runs):            
                    m_vals = [] 
                    for phcm in phcms:    
                        m_vals.append(
                            round(arch_dict[f"{lrn}.{phcm}"][i][measure],5)
                        )    
                    m_vals = list(zip(phcms,m_vals))
                    ranking = rank_list(m_vals, desc)            
                    
                    for name, rank in ranking:
                        rank_vec = ranking_dict[ds_name][measure][lrn][name]
                        rank_vec[rank-1]  += 1
                        ranking_dict[ds_name][measure][lrn][name] = rank_vec 

    return ranking_dict


def rank_list(values:list[list[str,float]], desc:bool):
    ranking = []
    for name, value in values:
        rank = 1
        for n, v in values:
            if n != name:
                if desc: #Higher is better 
                    rank += 1 if v > value else 0
                else: #Lower is better
                    rank += 1 if v < value else 0
        ranking.append([name, rank])
    return ranking 


def rank_across_runs_by_me(agg_key:str, ranking_m:dict, ranking_dict:dict, archs:list, n_archs:int) -> dict:
    """
    For each measure aggregate ranks across datasets to get ranking vectors per measure and arch.
    The  vectors reflect the counts of the arch having a given ranking.

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
    ranking_dict[agg_key] = {k:{a:np.zeros(n_archs) for a in archs} for k in ranking_m.keys()}
    for ds_name, ms_dict in ranking_dict.items():    
        
        if ds_name == agg_key:
            continue

        for measure, m_dict in ms_dict.items():
            for arch_name, ds_rank in m_dict.items():
                ranking_dict[agg_key][measure][arch_name] += ds_rank 
    
    return ranking_dict

def delta_rank_across_runs_by_me_lr(agg_key:str, ranking_m:dict, ranking_dict:dict, archs:list) -> dict:
    """
    For each measure aggregate ranks across datasets to get ranking vectors per measure and model.
    The vectors reflect the counts of the phcm having a given ranking.

    Args:
        agg_key (str): The key to store the results under.
        ranking_m (dict): See: def rank_across_runs_by_ds_me_lr
        ranking_dict (dict): See: def rank_across_runs_by_ds_me_lr
        archs (list): See: def rank_across_runs_by_ds_me_lr

    Returns:
        dict: See: def rank_across_runs_by_ds_me_lr
            It's ranking_dict with a new outer key added {agg_key}
    """
    lrns = get_learners(archs)
    phcms = get_phcm(archs)
    ranking_dict[agg_key] = {m:{l:{p:np.zeros(len(phcms)) for p in phcms} for l in lrns} for m in ranking_m} 

    for ds_name, ms_dict in ranking_dict.items():        
        if ds_name == agg_key:
            continue
        for measure, m_dict in ms_dict.items():
            for lrn, phcms_dict in m_dict.items():
                for phcm in phcms:
                    ranking_dict[agg_key][measure][lrn][phcm] += phcms_dict[phcm] 
    return ranking_dict


def delta_rank_across_runs_by_me(agg_key:str, ranking_m:dict, ranking_dict:dict, archs:list) -> dict:
    """
    For each measure aggregate ranks across learners to get ranking vectors per measure.
    The vectors reflect the counts of the phcm having a given ranking.

    Args:
        agg_key (str): The key to store the results under.
        ranking_m (dict): See: def rank_across_runs_by_ds_me_lr
        ranking_dict (dict): See: def rank_across_runs_by_ds_me_lr
        archs (list): See: def rank_across_runs_by_ds_me_lr

    Returns:
        dict: See: def rank_across_runs_by_ds_me_lr
            It's ranking_dict with a new outer key added {agg_key}
    """
    phcms = get_phcm(archs)
    for measure, m_dict in ranking_dict[agg_key].items():
        ranking_dict[agg_key][measure][agg_key] = {p:np.zeros(len(phcms)) for p in phcms}
        
        for lrn, phcms_dict in m_dict.items():
            if lrn == agg_key:
                continue
            for phcm in phcms:
                ranking_dict[agg_key][measure][agg_key][phcm] += phcms_dict[phcm] 
    return ranking_dict


def delta_rank_across_runs(agg_key:str, ranking_dict:dict, archs:list) -> dict:
    """
    Aggregate ranks across measures.
    The vectors reflect the counts of the phcm having a given ranking.

    Args:
        agg_key (str): The key to store the results under.
        ranking_m (dict): See: def rank_across_runs_by_ds_me_lr
        ranking_dict (dict): See: def rank_across_runs_by_ds_me_lr
        archs (list): See: def rank_across_runs_by_ds_me_lr

    Returns:
        dict: See: def rank_across_runs_by_ds_me_lr
            It's ranking_dict with a new outer key added {agg_key}
    """
    phcms = get_phcm(archs)
    ranking_dict[agg_key][agg_key] = {}
    ranking_dict[agg_key][agg_key][agg_key] = {p:np.zeros(len(phcms)) for p in phcms}
    for measure, m_dict in ranking_dict[agg_key].items():
        if measure == agg_key:
            continue
        for phcm in phcms:
            ranking_dict[agg_key][agg_key][agg_key][phcm] += m_dict[agg_key][phcm] 
    return ranking_dict


def rank_across_runs(agg_key:str, ranking_dict:dict, archs:list, n_archs:int) -> dict:
    """
    Aggregate across mesures to get ranking vectors per arch.
    The ranking vectors reflect the count of the arch having a given ranking 
    among all runs,datasets and measures.

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
            ranking_dict[agg_key][agg_key][arch_name] += m_rank 
            
    return ranking_dict


def calc_expected_ranking(ranking_dict:dict) -> dict:
    """Transforms all the count vectors into scalars by calculating expectation. 

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
            for arch_name, count_vec in ms_dict.items():
                sum = np.sum(count_vec)
                prob_vec = count_vec / sum
                ranks = np.arange(1, len(prob_vec) + 1)    
                exp_rank = np.dot(ranks, prob_vec)
                exp_ranking_dict[ds_name][ms_name][arch_name] = exp_rank
    return exp_ranking_dict

def calc_expected_delta_ranking(ranking_dict:dict) -> dict:
    """Transforms all the count vectors into scalars by calculating expectation. 

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
            for lrn, phcms in ms_dict.items():
                exp_ranking_dict[ds_name][ms_name][lrn] = {}
                for phcm, count_vec in phcms.items():
                    sum = np.sum(count_vec)
                    prob_vec = count_vec / sum
                    ranks = np.arange(1, len(prob_vec) + 1)    
                    exp_rank = np.dot(ranks, prob_vec)
                    exp_ranking_dict[ds_name][ms_name][lrn][phcm] = exp_rank
    return exp_ranking_dict
