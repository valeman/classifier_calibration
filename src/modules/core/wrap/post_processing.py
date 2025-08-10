from modules.core.wrap.wrappers import PostProcessing
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from betacal import BetaCalibration
from venn_abers import VennAbersCalibrator
import pearsonify as pear
import numpy as np 
import pandas as pd


class WrapPearsonify:
    """
    The Pearsonify class wraps the post-hoc calibration technique from Pearsonify to provide a standard API
    """
    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        self.q_alpha = None
        
    def fit(self,y_instance:np.array, y_target:pd.Series) -> None:
        y_target = np.asarray(y_target)
        y_instance = y_instance.reshape(-1,)  
        y_target = y_target.reshape(-1,)
        y_instance = np.clip(y_instance, a_min=1e-4, a_max=1 - 1e-4)
        residuals = pear.utils.compute_pearson_residuals(y_target, y_instance)
        self.q_alpha = np.quantile(np.abs(residuals), 1 - self.alpha)
        

    def predict_proba(self, y:np.array) -> np.array:
        y = y.reshape(-1,)
        lower_bounds, upper_bounds = pear.utils.compute_confidence_intervals(
            y, self.q_alpha
        )
        y_prob = upper_bounds / (1 - lower_bounds + upper_bounds)
        return y_prob.reshape(-1,)


def get_pps(suite:str, random_seed:int=123, n_cores:int=-1):
    match suite:
        case "v.1":
            pps = get_v1(random_seed, n_cores)
        case _:
            raise NotImplementedError
    return pps


def get_v1(SEED:int, n_cores:int=-1):
    """
    "none": No post processing
    "platt": Platt scaling
    "isotonic": Isotonic regression
    "beta": Beta calibration
    "venn_abers": Venn-abers 
    "pearsonify": Pearsonify
    """
    pps = [None]
    pp_std_fit = lambda learner, y_in, y_ta: learner.fit(y_in, y_ta)
    pp_std_predict_prob = lambda learner, y: learner.predict_proba(y)
    pp_std_predict = lambda learner, y: learner.predict(y)
    
    lr_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":n_cores}
    platt_scaling = PostProcessing(
        pp_name="platt"
        ,pp_class = LogisticRegression
        ,instatiator_fn = lr_instantiator
        ,fit_fn = pp_std_fit
        ,predict_prob_fn = pp_std_predict_prob
    )
    pps.append(platt_scaling)
    
    ir_instantiator = lambda meta_data: {"out_of_bounds":"clip"}
    isotonic_regression = PostProcessing(
        pp_name="isotonic"
        ,pp_class = IsotonicRegression
        ,instatiator_fn = ir_instantiator
        ,fit_fn = pp_std_fit
        ,predict_prob_fn = pp_std_predict
    )
    pps.append(isotonic_regression)

    bc_instantiator = lambda meta_data: {"parameters":"abm"}
    beta_calibration = PostProcessing(
        pp_name="beta"
        ,pp_class = BetaCalibration
        ,instatiator_fn = bc_instantiator
        ,fit_fn = pp_std_fit
        ,predict_prob_fn = pp_std_predict
    )
    pps.append(beta_calibration)

    class pp_va_fns:
        def __init__(self):
            self.y_in = None
            self.y_ta = None
        
        def fit(self, learner:VennAbersCalibrator, y_in:np.array ,y_ta:pd.Series) -> None:
            y_in = np.concatenate([1 - y_in, y_in], axis=1)
            self.y_in = y_in
            self.y_ta = np.asarray(y_ta)

        def predict_proba(self, learner:VennAbersCalibrator, y:np.array):
            y = np.concatenate([1 - y, y], axis=1)
            return learner.predict_proba(p_cal=self.y_in, y_cal=self.y_ta, p_test=y) 
    
    va_instantiator = lambda meta_data: {"random_state":SEED}    
    pp_va_fn = pp_va_fns()
    #Referred to as manual Venn-ABERS calibration and Pre-fitted Venn-ABERS calibration in the docs
    venn_abers_calibration = PostProcessing(
        pp_name="venn_abers"
        ,pp_class = VennAbersCalibrator
        ,instatiator_fn = va_instantiator
        ,fit_fn = pp_va_fn.fit
        ,predict_prob_fn = pp_va_fn.predict_proba
    )
    pps.append(venn_abers_calibration)
    
    pe_instantiator = lambda meta_data: {"alpha":0.05}
    pearsonify = PostProcessing(
        pp_name="pearsonify"
        ,pp_class = WrapPearsonify
        ,instatiator_fn = pe_instantiator
        ,fit_fn = pp_std_fit
        ,predict_prob_fn = pp_std_predict_prob
    )
    pps.append(pearsonify)
    return pps
