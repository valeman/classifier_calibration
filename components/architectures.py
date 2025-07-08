import components.config as config 
import sklearn 
import xgboost
import catboost
import lightgbm
import keras
import tabpfn
import betacal
import venn_abers
import pearsonify



SEED = config.SEED

class post_processing:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, y_in, y_out):
        raise NotImplementedError
    
    def predict_proba(self, y):
        raise NotImplementedError


class Model:
    def __init__(self, learner, fit_fn, predict_proba_fn):
        self.learner = learner
        self._fit_fn = fit_fn
        self._predict_proba_fn = predict_proba_fn
    
    def fit(self, x, y):
        return self._fit_fn(self.learner, x, y)
    
    def predict_proba(self, x):
        return self._predict_proba_fn(self.learner, x)


class Architecture:
    def __init__(self, name, model, calibration_set = False, post_processing=None):
        self.name = name
        self.model = model

        self.calibration_set = calibration_set
        self.post_processing = post_processing
        
    def fit(self, x_train, y_train, x_calibration=None, y_calibration=None):
        
        self.model.train(x_train, y_train)

        if self.post_processing:
            if self.calibration_set:
                y_cal_proba = self.model.predict_proba(x_calibration)
                self.post_processing.train(y_cal_proba, y_calibration)
            else:
                y_cal_proba = self.model.predict_proba(x_train)
                self.post_processing.train(y_cal_proba, y_train)


    def predict_proba(self, x):
        y_proba = self.model.predict(x)
        
        if self.post_processing:
            y_proba = self.post_processing.predict_proba(y_proba)
        
        return y_proba



    def predict(self, x):
        raise NotImplementedError


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
        std_fit = lambda learner, x, y: learner.train(x, y)
        std_predict_proba = lambda learner, x: learner.predict(x)

        cb = Model(learner=catboost.CatBoostClassifier()
                   ,fit_fn=std_fit
                   ,predict_proba_fn=std_predict_proba 
        ) 

        archs.append(Architecture(name="catboost" ,model=cb))
        
        self.architectures = archs