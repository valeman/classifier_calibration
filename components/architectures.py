import components.config as cf
import numpy as np 

from xgboost import XGBClassifier
import catboost
from lightgbm import LGBMClassifier
import keras
from tabpfn import TabPFNClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from betacal import BetaCalibration
from venn_abers import VennAbersCalibrator
import pearsonify as pear

SEED = cf.SEED

class Pearsonify:
    def __init__(self, alpha = 0.05):
        self.alpha = alpha
        self.q_alpha = None
        
    def fit(self,y_instance, y_target):
        y_target = np.asarray(y_target)
        residuals = pear.utils.compute_pearson_residuals(y_target, y_instance)
        self.q_alpha = np.quantile(np.abs(residuals), 1 - self.alpha)
        
    def predict_proba(self, y):
        lower_bounds, upper_bounds = pear.utils.compute_confidence_intervals(
            y, self.q_alpha
        )
        y_prob = upper_bounds / (1 - lower_bounds + upper_bounds)
        return y_prob.reshape(-1,)


class PreProcessing:
    def __init__(self):
        raise NotImplementedError
    
    def apply(self, x):
        raise NotImplementedError


class PostProcessing:
    def __init__(self, learner_class,instatiator_fn, fit_fn, predict_prob_fn):
        self.learner_class = learner_class
        self.instatiator_fn = instatiator_fn
        self.learner = None
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn
    
    def instantiate(self, meta_data:dict):
        cf.logger.info(f"Post-processing instantiated with:{self.instatiator_fn(meta_data)}")
        self.learner = self.learner_class(**self.instatiator_fn(meta_data))

    def fit(self, y_instance, y_target):
        self._fit_fn(self.learner, y_instance, y_target)
    
    def predict_prob(self, y):
        return self._predict_prob_fn(self.learner, y)


class Model:
    def __init__(self, learner_class,instatiator_fn, fit_fn, predict_prob_fn):
        self.learner_class = learner_class
        self.instatiator_fn = instatiator_fn
        self.learner = None
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn
    
    def instantiate(self, meta_data:dict):
        cf.logger.info(f"Model instantiated with:{self.instatiator_fn(meta_data)}")
        self.learner = self.learner_class(**self.instatiator_fn(meta_data))

    def fit(self, x, y):
        self._fit_fn(self.learner, x, y)
    
    def predict_prob(self, x):
        return self._predict_prob_fn(self.learner, x)


class Architecture:
    def __init__(self
                 ,name:str
                 ,model:Model
                 ,pre_trained_model=False
                 ,calibration_set=False
                 ,post_processing:PostProcessing=None
        ):
        self.name = name
        self.model = model
        self.pre_trained_model = pre_trained_model

        self.calibration_set = calibration_set
        self.post_processing = post_processing
        

    def fit(self, meta_data, x_train, y_train, x_calibration=None, y_calibration=None):
        
        self.model.instantiate(meta_data)
        self.model.fit(x_train, y_train)

        if not (self.post_processing is None):
            y_cal_prob = self.model.predict_prob(x_calibration if self.calibration_set else x_train)
            
            y_cal_prob = np.asarray(y_cal_prob)
            
            if y_cal_prob.ndim == 2 and y_cal_prob.shape[1] == 2:
                y_cal_prob =  y_cal_prob[:, 1]
            
            if y_cal_prob.ndim == 1:
                y_cal_prob = y_cal_prob.reshape(-1,1)

            self.post_processing.instantiate(meta_data)
            self.post_processing.fit(y_cal_prob, y_calibration if self.calibration_set else y_train)


    def predict_prob(self, x):
        y_prob = self.model.predict_prob(x)
        y_prob = np.asarray(y_prob)

        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob =  y_prob[:, 1]
    
        if self.post_processing:
            if y_prob.ndim == 1:
                y_prob = y_prob.reshape(-1,1)

            y_prob = self.post_processing.predict_prob(y_prob)
            
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob =  y_prob[:, 1]

        return y_prob
    

    def predict(self, x=None, y_prob=None):
        if x:
            y_prob = self.predict_prob(x)
        
        if y_prob.ndim == 1:
            return np.where(y_prob >= 0.5, 1, 0)
        else:
            return np.argmax(y_prob, axis=1) + 1


class ArchitectureSuite:
    def __init__(self, suite_name:str):
        self.suite_name = suite_name
        self.architectures = []

        match suite_name:
            case "v.1":
                self.init_v1()
    
    def __iter__(self) -> Architecture:
        for architecture in self.architectures:
            yield architecture
    

    def init_v1(self):
        archs = []

        md_std_fit = lambda learner, x, y: learner.fit(x, y)
        md_std_predict_prob = lambda learner, x: learner.predict_proba(x)
        
        cb_instantiator = lambda meta_data: {"random_seed":SEED, "verbose":False, "cat_features":meta_data["cat_features"]}
        cb = Model(learner_class=catboost.CatBoostClassifier
                   ,instatiator_fn=cb_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
        
        rf_instantiator = lambda meta_data: {"random_state":SEED}
        rf = Model(learner_class=RandomForestClassifier
                   ,instatiator_fn=rf_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        
        xgb_instantiator = lambda meta_data: {"random_state":SEED, "enable_categorical":True}
        xgb = Model(learner_class=XGBClassifier
                   ,instatiator_fn=xgb_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
              
        lgbm_instantiator = lambda meta_data: {"random_state":SEED}
        lgbm = Model(learner_class=LGBMClassifier
                   ,instatiator_fn=lgbm_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        lr_instantiator = lambda meta_data: {"random_state":SEED}
        lr = Model(learner_class=LogisticRegression
                   ,instatiator_fn=lr_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        
        knn_instantiator = lambda meta_data: {}
        knn = Model(learner_class=KNeighborsClassifier
                   ,instatiator_fn=knn_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        # svm_instantiator = lambda meta_data: {"probability":True, "random_state":SEED}
        # svm = Model(learner_class=SVC
        #            ,instatiator_fn=svm_instantiator
        #            ,fit_fn=md_std_fit
        #            ,predict_prob_fn=md_std_predict_prob
        # ) 

        
        mlp_instantiator = lambda meta_data: {"random_state":SEED}
        mlp = Model(learner_class=MLPClassifier
                   ,instatiator_fn=mlp_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        tpfn_instantiator = lambda meta_data: {"random_state":SEED, "ignore_pretraining_limits":True, "memory_saving_mode":True, "fit_mode":"low_memory"}
        tpfn = Model(learner_class=TabPFNClassifier
                   ,instatiator_fn=tpfn_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
        

        pp_std_fit = lambda learner, y_in, y_ta: learner.fit(y_in, y_ta)
        pp_std_predict_prob = lambda learner, y: learner.predict_proba(y)
        pp_std_predict = lambda learner, y: learner.predict(y)
        
        platt_scaling = PostProcessing(
            learner_class = LogisticRegression
            ,instatiator_fn = lr_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict_prob
        )
        
        ir_instantiator = lambda meta_data: {"out_of_bounds":"clip"}
        isotonic_regression = PostProcessing(
            learner_class = IsotonicRegression
            ,instatiator_fn = ir_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict
        )
        
        bc_instantiator = lambda meta_data: {"parameters":"abm"}
        beta_calibration = PostProcessing(
            learner_class = BetaCalibration
            ,instatiator_fn = bc_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict
        )
        
        class pp_va_fns:
            def __init__(self):
                self.y_in = None
                self.y_ta = None
            def fit(self,learner,y_in,y_ta):
                y_in = y_in.reshape(-1, 1)
                y_in = np.concatenate([1 - y_in, y_in], axis=1)
                self.y_in = y_in
                self.y_ta = np.asarray(y_ta)
            def predict_proba(self,learner,y):
                y = y.reshape(-1, 1)
                y = np.concatenate([1 - y, y], axis=1)
                return learner.predict_proba(p_cal=self.y_in, y_cal=self.y_ta, p_test=y) 
        
        va_instantiator = lambda meta_data: {}    
        pp_va_fn = pp_va_fns()
        #Referred to as manual Venn-ABERS calibration in the docs
        venn_abers_calibration = PostProcessing(
            learner_class = VennAbersCalibrator
            ,instatiator_fn = va_instantiator
            ,fit_fn = pp_va_fn.fit
            ,predict_prob_fn = pp_va_fn.predict_proba
        )
        
        pe_instantiator = lambda meta_data: {"alpha":0.05}
        pearsonify = PostProcessing(
            learner_class = Pearsonify
            ,instatiator_fn = pe_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict_prob
        )
        
        #archs.append(Architecture(name="catboost" ,model=cb))
        #archs.append(Architecture(name="random_forest" ,model=rf))
        #archs.append(Architecture(name="xgboost" ,model=xgb))
        #archs.append(Architecture(name="lightgbm" ,model=lgbm))
        #archs.append(Architecture(name="logistic_regression" ,model=lr))
        #archs.append(Architecture(name="k_nn" ,model=knn))
        #archs.append(Architecture(name="mlp" ,model=mlp))
        #archs.append(Architecture(name="tabpfn" ,model=tpfn))
       

        # archs.append(Architecture(
        #          name="catboost.platt_scaling"
        #          ,model=cb
        #          ,calibration_set=True
        #          ,post_processing=platt_scaling
        # ))

        # archs.append(Architecture(
        #          name="catboost.isotonic_regression"
        #          ,model=cb
        #          ,calibration_set=True
        #          ,post_processing=isotonic_regression
        # ))

        # archs.append(Architecture(
        #          name="catboost.beta_calibration"
        #          ,model=cb
        #          ,calibration_set=True
        #          ,post_processing=beta_calibration
        # ))
        
        # archs.append(Architecture(
        #          name="catboost.venn_abers_calibration"
        #          ,model=cb
        #          ,calibration_set=True
        #          ,post_processing=venn_abers_calibration
        # ))

        # archs.append(Architecture(
        #          name="catboost.pearsonify"
        #          ,model=cb
        #          ,calibration_set=True
        #          ,post_processing=pearsonify
        # ))

        self.architectures = archs