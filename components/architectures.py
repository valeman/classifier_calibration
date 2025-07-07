import config 

SEED = config.SEED

class post_processing:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError


class model:
    def __init__(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError


class Architecture:
    def __init__(self):
        self.name = None
        self.calibration_set = None
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError


class ArchitectureSuite:
    def __init__(self, suite_name:str):
        raise NotImplementedError
    
    def __iter__(self) -> Architecture:
        raise NotImplementedError