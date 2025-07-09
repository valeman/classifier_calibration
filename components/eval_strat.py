import components.config as config 
import components.data as data
import components.architectures as archs
import components.performance as perf

SEED = config.SEED


class EvaluationStrategy:
    def __init__(self, strategy:str, dataset:data.Dataset, arch:archs.Architecture,  p_measures:perf.PerformanceMeasures):
        self.strategy = strategy
        self.dataset = dataset
        self.arch = arch 
        self.p_measures = p_measures
        self.folds = None

        match strategy:
            case "v.1":
                self.init_v1(dataset, arch.calibration_set)

    def run(self) -> list[dict]:
        results = []
        for x_train, y_train, x_calibration, y_calibration, x_test, y_test in self.folds:
            
            self.arch.train(x_train, y_train, x_calibration, y_calibration)
            y_prob= self.arch.predict_prob(x_test)
            y_pred = self.arch.predict(x_test)

            perf_measures = self.p_measures.calc_perf(x_test, y_prob ,y_pred, y_test)
            results.append(perf_measures)
            
        return results
    
    def init_v1(self):
        raise NotImplementedError