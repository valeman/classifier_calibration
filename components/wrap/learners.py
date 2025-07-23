from components.wrap.wrappers import Learner
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from tabpfn import TabPFNClassifier 
import pandas as pd
import numpy as np


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


def get_learners(suite:str, random_seed:int=123):
    match suite:
        case "v.1":
            get_v1(random_seed)
        case _:
            raise NotImplementedError
    

def get_v1(SEED:int):
    learners = []
    md_std_fit = lambda learner, x, y: learner.fit(x, y)
    md_std_predict_prob = lambda learner, x: learner.predict_proba(x)
    
    cb_instantiator = lambda meta_data: {"random_seed":SEED
                                            ,"thread_count":-1
                                            ,"verbose":False
                                            ,"cat_features":meta_data["cat_features"]}
    cb = Learner(learner_name="cb"
                ,learner_class= CatBoostClassifier
                ,instatiator_fn=cb_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    ) 
    learners.append(cb)
    
    rf_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
    rf = Learner(learner_name="rf"
                ,learner_class=RandomForestClassifier
                ,instatiator_fn=rf_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    )
    learners.append(rf) 

    
    xgb_instantiator = lambda meta_data: {"random_state":SEED, "enable_categorical":True, "n_jobs":-1}
    xgb = Learner(learner_name="xgb"
                ,learner_class=XGBClassifier
                ,instatiator_fn=xgb_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    )
    learners.append(xgb) 
    
    md_lgbm_fit = lambda learner, x, y: learner.fit(x, y
                                                    , categorical_feature=x.select_dtypes(include='category').columns.tolist()
                                                    )
    lgbm_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
    lgbm = Learner(model_name="lgbm"
                ,learner_class=LGBMClassifier
                ,instatiator_fn=lgbm_instantiator
                ,fit_fn=md_lgbm_fit
                ,predict_prob_fn=md_std_predict_prob
    )
    learners.append(lgbm) 

    lr_instantiator = lambda meta_data: {"random_state":SEED, "n_jobs":-1}
    md_lr_fit = lambda learner, x, y: learner.fit(x[x.columns].astype("float"), y)
    md_lr_predict_prob = lambda learner, x: learner.predict_proba(x[x.columns].astype("float"))
    lr = Learner(model_name="lr"
                ,learner_class=LogisticRegression
                ,instatiator_fn=lr_instantiator
                ,fit_fn=md_lr_fit
                ,predict_prob_fn=md_lr_predict_prob
    ) 
    learners.append(lr)
    
    knn_instantiator = lambda meta_data: {"n_jobs":-1}
    knn = Learner(model_name="knn"
                ,learner_class=KNeighborsClassifier
                ,instatiator_fn=knn_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    ) 
    learners.append(knn)
    
    svm_instantiator = lambda meta_data: {"probability":True, "random_state":SEED}
    svm = Learner(model_name="svm"
                ,learner_class=SVC
                ,instatiator_fn=svm_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    )
    learners.append(svm)

    md_mlp_fit = lambda learner, x, y: learner.fit(x[x.columns].astype("float"), y)
    md_mlp_predict_prob = lambda learner, x: learner.predict_proba(x[x.columns].astype("float"))
    mlp_instantiator = lambda meta_data: {"random_state":SEED}
    mlp = Learner(model_name="mlp"
                ,learner_class=MLPClassifier
                ,instatiator_fn=mlp_instantiator
                ,fit_fn=md_mlp_fit
                ,predict_prob_fn=md_mlp_predict_prob
    ) 
    learners.append(mlp)

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
    
    ttra = Learner(model_name="ttra"
                ,learner_class=WrapTabTransformer
                ,instatiator_fn=ttra_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
    )
    learners.append(ttra) 


    tpfn_instantiator = lambda meta_data: {
        "random_state":SEED,
        "categorical_features_indices":meta_data["cat_features_indices"],
        "ignore_pretraining_limits":True,
        "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
        "fit_mode":"low_memory",
        "memory_saving_mode":"auto",
        "n_jobs":-1
    }
    tpfn = Learner(model_name="tabpfn"
                ,learner_class=TabPFNClassifier
                ,instatiator_fn=tpfn_instantiator
                ,fit_fn=md_std_fit
                ,predict_prob_fn=md_std_predict_prob
                ,pre_trained_model=True
    ) 
    learners.append(tpfn)

    return learners
