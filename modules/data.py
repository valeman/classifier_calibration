import openml

class Dataset:
    def __init__(self, df, source, features, target, df_type, meta_data):
        self.df = df
        self.source = source
        self.df_type = df_type
        self.features = features
        self.target = target
        self.meta_data = meta_data
        
    def to_pandas(self):
        pass


class DatasetSuite:
    def __init__(self, suite):
        self.suite = suite
        self.dataset_suite = None

        match suite:
            case "Tabarena-v0.1":
                self.dataset_suite = self.get_tabarena_v01()
            case _:
                raise NotImplementedError

    def __iter__(self):
        for dataset in self.dataset_suite:
            yield dataset
        
    def get_tabarena_v01(self):
        raise NotImplementedError
    
