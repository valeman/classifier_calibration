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
    def __init__(self, suite_name):
        self.suite_name = suite_name
        self.source = None
        self.dataset_suite = None
        self.n_datasets = None

        match suite_name:
            case "Tabarena-v0.1":
                self.dataset_suite = self.get_tabarena_v01()
            case _:
                raise NotImplementedError

    def __iter__(self):
        for i in range(self.n_datasets):
            match self.source:
                case "openml":
                    pass
                case _:
                    raise NotImplementedError
        
    def get_tabarena_v01(self):
        suite = openml.study.get_suite(457) #Study: TabArena-v0.1 Suite
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        class_tasks = tasks[tasks["task_type"] == "Supervised Classification"]
        class_tasks = class_tasks[["tid", "name" ,  "target_feature"]]
        class_tasks = class_tasks.reset_index(drop=True)
        self.n_datasets = len(class_tasks)
        self.source = "openml"
        return class_tasks
