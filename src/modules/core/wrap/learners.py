from modules.core.wrap.wrappers import Learner
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from tabpfn import TabPFNClassifier
from autogluon.common.features.feature_metadata import FeatureMetadata
from tabrepo.benchmark.models.ag import (
    ModernNCAModel,
    TabMModel,
    TabICLModel,
    RealMLPModel,
)
from interpret.glassbox import ExplainableBoostingClassifier
import pandas as pd
import numpy as np


class WrapTabTransformer:
    """
    The WrapTabTransformer class wraps the TabTransformer implementation of pytorch-tabular to provide a standard API
    """

    def __init__(
        self,
        num_workers: int,
        random_state: int,
        continuous_cols: list[str],
        categorical_cols: list[str],
        auto_lr_find: bool = True,
        batch_size: int = 1024,
        max_epochs: int = 20,
        devices: int = -1,
        verbose: bool = False,
    ):
        self.random_state = random_state
        model_config = TabTransformerConfig(task="classification", seed=random_state)
        data_config = DataConfig(
            target=["target"],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            num_workers=num_workers,
        )
        trainer_config = TrainerConfig(
            auto_lr_find=auto_lr_find,
            batch_size=batch_size,
            max_epochs=max_epochs,
            devices=devices,
            progress_bar="none",
            seed=random_state,
        )
        optimizer_config = OptimizerConfig()

        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=verbose,
            suppress_lightning_logger=not verbose,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        df = pd.concat([x, y.rename("target")], axis=1)
        df[df.columns] = df[df.columns].astype("object")
        self.tabular_model.fit(train=df, seed=self.random_state)

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        x[x.columns] = x[x.columns].astype("object")
        preds = self.tabular_model.predict(x)
        preds = np.asarray(preds["target_1_probability"]).reshape(
            -1,
        )
        return preds


class WrapTabRepoModels:
    """
    The WrapTabRepoModels class wraps models from tabrepo/autogluon to provide a standard API
    """

    def __init__(
        self,
        model: str,
        continuous_cols: list[str],
        categorical_cols: list[str],
        n_cores: int = -1,
    ):
        self.n_cores = n_cores
        ftypes = {}
        ftypes.update({c: "category" for c in categorical_cols})
        ftypes.update({c: "float" for c in continuous_cols})
        self.feature_md = FeatureMetadata(type_map_raw=ftypes)

        match model:
            case "nca":
                self.clf = ModernNCAModel(problem_type="binary")
            case "tabm":
                self.clf = TabMModel(problem_type="binary")
            case "ticl":
                self.clf = TabICLModel(problem_type="binary")
            case "remlp":
                self.clf = RealMLPModel(problem_type="binary")
            case _:
                raise NotImplementedError

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(
            X=x,
            y=y,
            num_cpus=self.n_cores,
            feature_metadata=self.feature_md,
            time_limit=60 * 30,  # 1/2 hour
            verbosity=1,
        )

    def predict_proba(self, x: pd.DataFrame) -> np.array:
        return self.clf.predict_proba(X=x)


def get_learners(suite: str, random_seed: int = 123, n_cores: int = -1):
    match suite:
        case "v.1":
            learners = get_v1(random_seed, n_cores)
        case _:
            raise NotImplementedError
    return learners


def get_v1(SEED: int, n_cores: int = -1):
    """
    "svm": Support vector machine
    "lr": Logistic Regression
    "lda": Linear Discriminant Analysis
    "knn": K-Nearest Neighbours
    "rf": RandomForest
    "ext": ExtraTrees
    "ebm": Explainable Boosting Machine
    "cb": Catboost
    "xgb": XGBoost
    "lgbm": LightGBM
    "nca": ModernNCA
    "ttra": TabTransformer
    "ticl": TabICL
    "tpfn": TabPFN
    "tabm": TabM
    "mlp": Multilayer Perceptron
    "remlp": Real Multilayer Perceptron
    """
    learners = []
    md_std_fit = lambda learner, x, y: learner.fit(x, y)
    md_std_predict_prob = lambda learner, x: learner.predict_proba(x)

    ext_instantiator = lambda meta_data: {"random_state": SEED, "n_jobs": n_cores}
    ext = Learner(
        learner_name="ext",
        learner_class=ExtraTreesClassifier,
        instatiator_fn=ext_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(ext)

    ebm_instantiator = lambda meta_data: {"n_jobs": n_cores, "random_state": SEED}
    md_ebm_fit = lambda learner, x, y: learner.fit(x.to_numpy(), y.to_numpy())
    md_ebm_predict_prob = lambda learner, x: learner.predict_proba(x.to_numpy())
    ebm = Learner(
        learner_name="ebm",
        learner_class=ExplainableBoostingClassifier,
        instatiator_fn=ebm_instantiator,
        fit_fn=md_ebm_fit,
        predict_prob_fn=md_ebm_predict_prob,
    )
    learners.append(ebm)

    cb_instantiator = lambda meta_data: {
        "random_seed": SEED,
        "thread_count": n_cores,
        "verbose": False,
        "cat_features": meta_data["cat_features"],
    }
    cb = Learner(
        learner_name="cb",
        learner_class=CatBoostClassifier,
        instatiator_fn=cb_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(cb)

    rf_instantiator = lambda meta_data: {"random_state": SEED, "n_jobs": n_cores}
    rf = Learner(
        learner_name="rf",
        learner_class=RandomForestClassifier,
        instatiator_fn=rf_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(rf)

    xgb_instantiator = lambda meta_data: {
        "random_state": SEED,
        "enable_categorical": True,
        "n_jobs": n_cores,
    }
    xgb = Learner(
        learner_name="xgb",
        learner_class=XGBClassifier,
        instatiator_fn=xgb_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(xgb)

    md_lgbm_fit = lambda learner, x, y: learner.fit(
        x, y, categorical_feature=x.select_dtypes(include="category").columns.tolist()
    )
    lgbm_instantiator = lambda meta_data: {"random_state": SEED, "n_jobs": n_cores}
    lgbm = Learner(
        learner_name="lgbm",
        learner_class=LGBMClassifier,
        instatiator_fn=lgbm_instantiator,
        fit_fn=md_lgbm_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(lgbm)

    lr_instantiator = lambda meta_data: {"random_state": SEED, "n_jobs": n_cores}
    md_lr_fit = lambda learner, x, y: learner.fit(x[x.columns].astype("float"), y)
    md_lr_predict_prob = lambda learner, x: learner.predict_proba(
        x[x.columns].astype("float")
    )
    lr = Learner(
        learner_name="lr",
        learner_class=LogisticRegression,
        instatiator_fn=lr_instantiator,
        fit_fn=md_lr_fit,
        predict_prob_fn=md_lr_predict_prob,
    )
    learners.append(lr)

    knn_instantiator = lambda meta_data: {"n_jobs": n_cores}
    knn = Learner(
        learner_name="knn",
        learner_class=KNeighborsClassifier,
        instatiator_fn=knn_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(knn)

    svm_instantiator = lambda meta_data: {"probability": True, "random_state": SEED}
    svm = Learner(
        learner_name="svm",
        learner_class=SVC,
        instatiator_fn=svm_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(svm)

    lda_instantiator = lambda meta_data: {}
    lda = Learner(
        learner_name="lda",
        learner_class=LinearDiscriminantAnalysis,
        instatiator_fn=lda_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(lda)

    

    nca_instantiator = lambda meta_data: {
        "model": "nca",
        "continuous_cols": meta_data["non_cat_features"],
        "categorical_cols": meta_data["cat_features"],
        "n_cores": n_cores,
    }
    nca = Learner(
        learner_name="nca",
        learner_class=WrapTabRepoModels,
        instatiator_fn=nca_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(nca)

    tabm_instantiator = lambda meta_data: {
        "model": "tabm",
        "continuous_cols": meta_data["non_cat_features"],
        "categorical_cols": meta_data["cat_features"],
        "n_cores": n_cores,
    }
    tabm = Learner(
        learner_name="tabm",
        learner_class=WrapTabRepoModels,
        instatiator_fn=tabm_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(tabm)

    ticl_instantiator = lambda meta_data: {
        "model": "ticl",
        "continuous_cols": meta_data["non_cat_features"],
        "categorical_cols": meta_data["cat_features"],
        "n_cores": n_cores,
    }

    ticl = Learner(
        learner_name="ticl",
        learner_class=WrapTabRepoModels,
        instatiator_fn=ticl_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(ticl)

    remlp_instantiator = lambda meta_data: {
        "model": "remlp",
        "continuous_cols": meta_data["non_cat_features"],
        "categorical_cols": meta_data["cat_features"],
        "n_cores": n_cores,
    }
    remlp = Learner(
        learner_name="remlp",
        learner_class=WrapTabRepoModels,
        instatiator_fn=remlp_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(remlp)

    md_mlp_fit = lambda learner, x, y: learner.fit(x[x.columns].astype("float"), y)
    md_mlp_predict_prob = lambda learner, x: learner.predict_proba(
        x[x.columns].astype("float")
    )
    mlp_instantiator = lambda meta_data: {"random_state": SEED}
    mlp = Learner(
        learner_name="mlp",
        learner_class=MLPClassifier,
        instatiator_fn=mlp_instantiator,
        fit_fn=md_mlp_fit,
        predict_prob_fn=md_mlp_predict_prob,
    )
    learners.append(mlp)

    ttra_instantiator = lambda meta_data: {
        "random_state": SEED,
        "num_workers": 10,
        "continuous_cols": meta_data["non_cat_features"],
        "categorical_cols": meta_data["cat_features"],
        "auto_lr_find": False,
        "batch_size": 1024,
        "max_epochs": 20,
        "devices": n_cores,
        "verbose": False,
    }

    ttra = Learner(
        learner_name="ttra",
        learner_class=WrapTabTransformer,
        instatiator_fn=ttra_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
    )
    learners.append(ttra)

    tpfn_instantiator = lambda meta_data: {
        "random_state": SEED,
        "categorical_features_indices": meta_data["cat_features_indices"],
        "ignore_pretraining_limits": True,
        "inference_config": {"SUBSAMPLE_SAMPLES": 10000},
        "fit_mode": "low_memory",
        "memory_saving_mode": "auto",
        "n_jobs": n_cores,
    }
    tpfn = Learner(
        learner_name="tabpfn",
        learner_class=TabPFNClassifier,
        instatiator_fn=tpfn_instantiator,
        fit_fn=md_std_fit,
        predict_prob_fn=md_std_predict_prob,
        pre_trained=True,
    )
    learners.append(tpfn)
    learners = [l for l in learners if l.learner_name in ["ticl"]]  # TODO: AMEND
    return learners
