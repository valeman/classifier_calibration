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
    
    def predict(self, y):
        raise NotImplementedError


class model:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self, x, y):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError


class Architecture:
    def __init__(self):
        self.name = None
        self.calibration_set = None
        self.learner = None
        self.post_calibration = None
        
    
    def fit(self, x_train, y_train, x_calibration=None, y_calibration=None):
        raise NotImplementedError
    
    def predict(self, x):
        raise NotImplementedError


class ArchitectureSuite:
    def __init__(self, suite_name:str):
        self.suite_name = suite_name
        self.architectures = None

        match suite_name:
            case "v.1":
                self.init_v1()
    
    def __iter__(self) -> Architecture:
        for architecture in self.architectures:
            yield architecture
    
    def init_v1(self):
        pass
