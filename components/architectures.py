import components.config as cf
import numpy as np 
import sklearn 
import xgboost
import catboost
import lightgbm
import keras
import tabpfn
import betacal
import venn_abers
import pearsonify



SEED = cf.SEED

class PreProcessing:
    def __init__(self):
        raise NotImplementedError
    
    def apply(self, x):
        raise NotImplementedError

class PostProcessing:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, y_in, y_out):
        raise NotImplementedError
    
    def predict_prob(self, y):
        raise NotImplementedError


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
        return self._fit_fn(self.learner, x, y)
    
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

        if self.post_processing:
            if self.calibration_set:
                y_cal_prob = self.model.predict_prob(x_calibration)
                self.post_processing.train(y_cal_prob, y_calibration)
            else:
                y_cal_prob = self.model.predict_prob(x_train)
                self.post_processing.train(y_cal_prob, y_train)


    def predict_prob(self, x):
        y_prob = self.model.predict_prob(x)
        y_prob = np.asarray(y_prob)

        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob =  y_prob[:, 1]
    
        if self.post_processing:
            y_prob = self.post_processing.predict_prob(y_prob)

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
        std_fit = lambda learner, x, y: learner.fit(x, y)
        std_predict_prob = lambda learner, x: learner.predict_proba(x)
        
        cb_instantiator = lambda meta_data: {"random_seed":SEED, "verbose":False, "cat_features":meta_data["cat_features"]}
        cb = Model(learner_class=catboost.CatBoostClassifier
                   ,instatiator_fn=cb_instantiator
                   ,fit_fn=std_fit
                   ,predict_prob_fn=std_predict_prob
        ) 
        archs.append(Architecture(name="catboost" ,model=cb))


        self.architectures = archs