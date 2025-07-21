from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from tabpfn import TabPFNClassifier 
from betacal import BetaCalibration
from venn_abers import VennAbersCalibrator
import pearsonify as pear
import logging 
import numpy as np 
import pandas as pd

lg = logging.getLogger(__name__)

class WrapTabTransformer:
    """
    The TabTransformer class wraps the TabTransformer implementation of pytorch-tabular to provide a standard API
    """
    def __init__(self, num_workers:int
            , random_state:int
            , continuous_cols:list[str]
            , categorical_cols:list[str]
            , auto_lr_find:bool=True
            , batch_size:int=1024
            , max_epochs:int=20
            , devices:int=-1
            , verbose:bool = False
            ):
        self.random_state = random_state
        model_config = TabTransformerConfig(
            task="classification"        
            ,seed=random_state
        )
        data_config = DataConfig(
            target=["target"],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            num_workers=num_workers
        )
        trainer_config = TrainerConfig(
            auto_lr_find=auto_lr_find,
            batch_size=batch_size,
            max_epochs=max_epochs,
            devices=devices,
        )
        optimizer_config = OptimizerConfig()
        
        self.tabular_model = TabularModel(
            data_config=data_config
            ,model_config=model_config
            ,optimizer_config=optimizer_config
            ,trainer_config=trainer_config
            ,verbose=verbose
        )
    
    def fit(self,x:pd.DataFrame,y:pd.Series) -> None:
        df = pd.concat([x, y.rename("target")], axis=1)
        df[df.columns] = df[df.columns].astype("object")
        self.tabular_model.fit(train=df, seed=self.random_state)
    
    def predict_proba(self,x:pd.DataFrame) -> np.array:
        x[x.columns] = x[x.columns].astype("object")
        preds = self.tabular_model.predict(x)
        preds = np.asarray(preds["target_1_probability"]).reshape(-1,)
        return preds       


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


class PreProcessing:
    """
    The PreProcessing class wraps around each pre processing technique to provide a standard API.  
    Only supports (X,Y) to (X,Y) maps.
    """
    def __init__(self):
        raise NotImplementedError
    
    def apply(self, x:pd.DataFrame,y:pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError
        return x, y


class PostProcessing:
    """
    The PostProcessing class wraps around each post processing technique to provide a standard API.  
    Only supports Y to Y maps.
    """
    def __init__(self,pp_name, learner_class,instatiator_fn, fit_fn, predict_prob_fn):
        self.pp_name = pp_name
        self.learner_class = learner_class
        self.instatiator_fn = instatiator_fn
        self.learner = None
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn
    
    def instantiate(self, meta_data:dict) -> None:
        lg.info(f"Post-processing:{self.pp_name} instantiated with: \n{self.instatiator_fn(meta_data)}")
        self.learner = self.learner_class(**self.instatiator_fn(meta_data))

    def fit(self, y_instance:np.array, y_target:pd.Series) -> None:
        self._fit_fn(self.learner, y_instance, y_target)
    
    def predict_prob(self, y:np.array) -> np.array:
        return np.asarray(self._predict_prob_fn(self.learner, y))


class Model:
    """
    The architecture class wraps around each architecture to provide a standard API.  
    """
    def __init__(self,model_name, learner_class,instatiator_fn, fit_fn, predict_prob_fn,pre_trained_model=False):
        self.model_name = model_name
        self.learner_class = learner_class
        self.instatiator_fn = instatiator_fn
        self.learner = None
        self.pre_trained_model = pre_trained_model
        self._fit_fn = fit_fn
        self._predict_prob_fn = predict_prob_fn
    
    def instantiate(self, meta_data:dict) -> None:
        lg.info(f"Model:{self.model_name} instantiated with:\n{self.instatiator_fn(meta_data)}")
        self.learner = self.learner_class(**self.instatiator_fn(meta_data))

    def fit(self, x:pd.DataFrame, y:pd.Series) -> None:
        self._fit_fn(self.learner, x, y)
    
    def predict_prob(self, x:pd.DataFrame) -> np.array:
        return np.asarray(self._predict_prob_fn(self.learner, x))


class Architecture:
    """
    The architecture class wraps around each architecture to provide a standard API.  
    """
    def __init__(self
                 ,name:str
                 ,model:Model
                 ,calibration_set=False
                 ,post_processing:PostProcessing=None
                 ,random_seed:int=123
        ):
        self.name = name
        self.model = model
        self.calibration_set = calibration_set
        self.post_processing = post_processing
        self.random_seed = random_seed
        
    def fit(self, meta_data:dict
            , x_train:pd.DataFrame
            , y_train:pd.Series
                ) -> None:
        
        x_calibration = None
        y_calibration = None
        if self.calibration_set:
            x_train, x_calibration, y_train, y_calibration = train_test_split(
                x_train
                ,y_train
                ,stratify=y_train
                ,shuffle=True
                ,train_size=0.8
                ,random_state=self.random_seed
            )

        self.model.instantiate(meta_data)
        self.model.fit(x_train, y_train)

        if not (self.post_processing is None):
            y_cal_prob = self.model.predict_prob(x_calibration if self.calibration_set else x_train)

            if y_cal_prob.ndim == 2 and y_cal_prob.shape[1] == 2:
                y_cal_prob =  y_cal_prob[:, 1]
            if y_cal_prob.ndim == 1:
                y_cal_prob = y_cal_prob.reshape(-1,1)

            self.post_processing.instantiate(meta_data)
            self.post_processing.fit(y_cal_prob, y_calibration if self.calibration_set else y_train)


    def predict_prob(self, x:pd.DataFrame) -> np.array:
        y_prob = self.model.predict_prob(x)

        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob =  y_prob[:, 1]
    
        if not (self.post_processing is None):
            if y_prob.ndim == 1:
                y_prob = y_prob.reshape(-1,1)

            y_prob = self.post_processing.predict_prob(y_prob)
            
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                y_prob =  y_prob[:, 1]

        return y_prob
    

    def predict(self, x=None, y_prob=None) -> np.array:
        if not (x is None):
            y_prob = self.predict_prob(x)
        
        if y_prob.ndim == 1:
            return np.where(y_prob >= 0.5, 1, 0)
        else:
            return np.argmax(y_prob, axis=1) + 1


class ArchitectureSuite:
    """
    The ArchitectureSuite class defines a collection of architectures (a suite) and returns each achitecture iteratively.
    All architectures are wrapped around the architecture class to provide a standard API. 
    """
    def __init__(self, suite_name:str, random_seed:int=123):
        self.suite_name = suite_name
        self.architectures = []
        self.n_architectures = None
        self.random_seed = random_seed
        match suite_name:
            case "v.1":
                self.init_v1()
    
    def __iter__(self) -> Architecture:
        for architecture in self.architectures:
            yield architecture
    

    def init_v1(self) -> None:
        """Defines the v.1 suite, which contains all combinations of the models:
            "svm": Support vector machine
            "lr": Logistic Regression
            "knn": K-Nearest Neighbours
            "rf": RandomForest
            "cb": Catboost
            "xgb": XGBoost
            "lgbm": LightGBM
            "ttra": TabTransformer
            "rftpfn": Randomforest TabPFN
            "mlp": Multilayer Perceptron 

            And the post-processing techniques:
            "none": No post processing
            "platt": Platt scaling
            "isotonic": Isotonic regression
            "beta": Beta calibration
            "venn_abers": Venn-abers 
            "pearsonify": Pearsonify
        """
        SEED = self.random_seed
        archs = []

        md_std_fit = lambda learner, x, y: learner.fit(x, y)
        md_std_predict_prob = lambda learner, x: learner.predict_proba(x)
        
        cb_instantiator = lambda meta_data: {"random_seed":SEED
                                             ,"thread_count":-1
                                             ,"verbose":False
                                             ,"cat_features":meta_data["cat_features"]}
        cb = Model(model_name="cb"
                   ,learner_class= CatBoostClassifier
                   ,instatiator_fn=cb_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
        
        rf_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
        rf = Model(model_name="rf"
                   ,learner_class=RandomForestClassifier
                   ,instatiator_fn=rf_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        
        xgb_instantiator = lambda meta_data: {"random_state":SEED, "enable_categorical":True, "n_jobs":-1}
        xgb = Model(model_name="xgb"
                   ,learner_class=XGBClassifier
                   ,instatiator_fn=xgb_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
        
        md_lgbm_fit = lambda learner, x, y: learner.fit(x, y
                                                        , categorical_feature=x.select_dtypes(include='category').columns.tolist()
                                                        )
        lgbm_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
        lgbm = Model(model_name="lgbm"
                   ,learner_class=LGBMClassifier
                   ,instatiator_fn=lgbm_instantiator
                   ,fit_fn=md_lgbm_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        lr_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
        md_lr_fit = lambda learner, x, y: learner.fit(x[x.columns].astype("float"), y)
        md_lr_predict_prob = lambda learner, x: learner.predict_proba(x[x.columns].astype("float"))
        lr = Model(model_name="lr"
                   ,learner_class=LogisticRegression
                   ,instatiator_fn=lr_instantiator
                   ,fit_fn=md_lr_fit
                   ,predict_prob_fn=md_lr_predict_prob
        ) 

        
        knn_instantiator = lambda meta_data: {"n_jobs":-1}
        knn = Model(model_name="knn"
                   ,learner_class=KNeighborsClassifier
                   ,instatiator_fn=knn_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 
        
        svm_instantiator = lambda meta_data: {"probability":True, "random_state":SEED}
        svm = Model(model_name="svm"
                   ,learner_class=SVC
                   ,instatiator_fn=svm_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        
        mlp_instantiator = lambda meta_data: {"random_state":SEED}
        mlp = Model(model_name="mlp"
                   ,learner_class=MLPClassifier
                   ,instatiator_fn=mlp_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 

        ttra_instantiator = lambda meta_data: {"random_state":SEED
                                                ,"num_workers":10
                                                , "continuous_cols":meta_data["non_cat_features"]
                                                , "categorical_cols":meta_data["cat_features"]
                                                , "auto_lr_find":False
                                                , "batch_size":1024
                                                , "max_epochs":20
                                                , "devices":-1
                                                , "verbose":False
        }
        
        ttra = Model(model_name="ttra"
                   ,learner_class=WrapTabTransformer
                   ,instatiator_fn=ttra_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
        ) 


        tpfn_instantiator = lambda meta_data: {
            "random_state":SEED,
            "categorical_features_indices":meta_data["cat_features_indices"],
            "ignore_pretraining_limits":True,
            "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
            "fit_mode":"low_memory",
            "memory_saving_mode":"auto",
            "n_jobs":-1
        }
        tpfn = Model(model_name="tabpfn"
                   ,learner_class=TabPFNClassifier
                   ,instatiator_fn=tpfn_instantiator
                   ,fit_fn=md_std_fit
                   ,predict_prob_fn=md_std_predict_prob
                   ,pre_trained_model=True
        ) 

        pp_std_fit = lambda learner, y_in, y_ta: learner.fit(y_in, y_ta)
        pp_std_predict_prob = lambda learner, y: learner.predict_proba(y)
        pp_std_predict = lambda learner, y: learner.predict(y)
        
        platt_scaling = PostProcessing(
            pp_name="platt"
            ,learner_class = LogisticRegression
            ,instatiator_fn = lr_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict_prob
        )
        
        ir_instantiator = lambda meta_data: {"out_of_bounds":"clip"}
        isotonic_regression = PostProcessing(
            pp_name="isotonic"
            ,learner_class = IsotonicRegression
            ,instatiator_fn = ir_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict
        )
        
        bc_instantiator = lambda meta_data: {"parameters":"abm"}
        beta_calibration = PostProcessing(
            pp_name="beta"
            ,learner_class = BetaCalibration
            ,instatiator_fn = bc_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict
        )
        
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
            ,learner_class = VennAbersCalibrator
            ,instatiator_fn = va_instantiator
            ,fit_fn = pp_va_fn.fit
            ,predict_prob_fn = pp_va_fn.predict_proba
        )
        
        pe_instantiator = lambda meta_data: {"alpha":0.05}
        pearsonify = PostProcessing(
            pp_name="pearsonify"
            ,learner_class = WrapPearsonify
            ,instatiator_fn = pe_instantiator
            ,fit_fn = pp_std_fit
            ,predict_prob_fn = pp_std_predict_prob
        )

        models = [
         svm,
         lr,
         knn,
         rf,
         cb,
         xgb,
         lgbm,
         ttra,
         tpfn,
         mlp
        ]
        
        phc = [
        None,
        platt_scaling,
        isotonic_regression,
        beta_calibration,
        venn_abers_calibration,
        pearsonify
        ]

        for m in models:
            for p in phc:
                if p is None:
                    archs.append(Architecture(name=m.model_name ,model=m, random_seed=SEED))
                else:
                    archs.append(Architecture(
                        name=f"{m.model_name}.{p.pp_name}"
                        ,model=m
                        ,calibration_set=True
                        ,post_processing=p
                        ,random_seed=SEED
                    ))
                    
        self.n_architectures = len(archs)
        self.architectures = archs