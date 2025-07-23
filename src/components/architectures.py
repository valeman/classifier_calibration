from components.wrap.wrappers import Learner, PostProcessing, PreProcessing
from components.wrap.learners import get_learners
from components.wrap.post_processing import get_pps
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Architecture:
    """
    The architecture class wraps around each architecture to provide a standard API.  
    """
    def __init__(self
                 ,name:str
                 ,learner:Learner
                 ,calibration_set=False
                 ,post_processing:PostProcessing=None
                 ,random_seed:int=123
        ):
        self.name = name
        self.learner = learner
        self.calibration_set = calibration_set
        self.post_processing = post_processing
        self.random_seed = random_seed
        
    def fit(self
            , meta_data:dict
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

        self.learner.instantiate(meta_data)
        self.learner.fit(x_train, y_train)

        if not (self.post_processing is None):
            y_cal_prob = self.learner.predict_prob(x_calibration if self.calibration_set else x_train)

            if y_cal_prob.ndim == 2 and y_cal_prob.shape[1] == 2:
                y_cal_prob =  y_cal_prob[:, 1]
            if y_cal_prob.ndim == 1:
                y_cal_prob = y_cal_prob.reshape(-1,1)

            self.post_processing.instantiate(meta_data)
            self.post_processing.fit(y_cal_prob, y_calibration if self.calibration_set else y_train)


    def predict_prob(self, x:pd.DataFrame) -> np.array:
        y_prob = self.learner.predict_prob(x)

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
        """Defines the v.1 suite, which contains all combinations of the learners:
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

        learners = get_learners(suite="v.1",random_seed=SEED)
        phc = get_pps(suite="v.1",random_seed=SEED)

        for l in learners:
            for p in phc:
                if p is None:
                    archs.append(Architecture(name=l.learner_name ,learner=l, random_seed=SEED))
                else:
                    archs.append(Architecture(
                        name=f"{l.learner_name}.{p.pp_name}"
                        ,learner=l
                        ,calibration_set=True
                        ,post_processing=p
                        ,random_seed=SEED
                    ))
                    
        self.n_architectures = len(archs)
        self.architectures = archs