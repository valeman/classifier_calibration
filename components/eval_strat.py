import config 
import data
import architectures as archs
import performance as perf

SEED = config.SEED


class EvaluationStrategy:
    def __init__(self, strategy:str, dataset:data.Dataset, arch:archs.Architecture,  p_measures:perf.PerformanceMeasures):
        self.strategy = strategy
        self.dataset = dataset
        self.arch = arch 
        self.p_measures = p_measures

    def run() -> dict:
        raise NotImplementedError

