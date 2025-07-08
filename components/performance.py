import components.config as config 
import pycaleva


class PerformanceMeasures:
    def __init__(self, suite_name:str):
        self.suite_name = suite_name
        self.computational_cost = False
        self.measure_functions = {}
        match suite_name:
            case "v1":
                self.computational_cost = True
                self.measure_functions = self.init_v1()
            case _:
                raise NotImplementedError
        
    def calc_perf(self, x, y_prob ,y_pred, y_test):
        pass

    def init_v1(self):
        pass