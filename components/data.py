import openml


class Dataset:
    def __init__(self, df, df_name, df_type, source, target, cat_columns, meta_data):
        self.df = df
        self.df_name = df_name
        self.df_type = df_type
        self.source = source
        self.target = target
        self.cat_columns = cat_columns
        self.meta_data = meta_data
        
    def to_pandas(self):
        match self.df_type:
            case "<class 'pandas.core.frame.DataFrame'>":
                return self.df
            
            case _:
                raise NotImplementedError


class DatasetSuite:
    def __init__(self, suite_name):
        self.suite_name = suite_name
        self.dataset_suite = None
        self.n_datasets = None

        match suite_name:
            case "Tabarena-v0.1":
                self.dataset_suite = self.get_tabarena_v01()
            
            case _:
                raise NotImplementedError


    def __iter__(self):
        for i in range(self.n_datasets):
            match self.suite_name:
                case "Tabarena-v0.1":
                    task_info = self.dataset_suite.iloc[i]
                    tid = int(task_info["tid"])
                    name = task_info["name"]
                    target = task_info["target_feature"]
                    task = openml.tasks.get_task(tid)
                    ds = task.get_dataset()
                    df, _, cat_features, at_names =  ds.get_data()
                    cat_columns = [i for i,j in zip(df.columns, cat_features) if j]
                    dataset = Dataset(df = df
                                    ,df_name = name
                                    ,df_type = str(type(df))
                                    ,source = "openml"
                                    ,target = target
                                    ,cat_columns = cat_columns
                                    ,meta_data = None
                    )
                case _:
                    raise NotImplementedError
            yield dataset

    def get_tabarena_v01(self):
        suite = openml.study.get_suite(457) #Study: TabArena-v0.1 Suite
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        class_tasks = tasks[tasks["task_type"] == "Supervised Classification"]
        class_tasks = class_tasks[["tid", "name" ,  "target_feature"]]
        class_tasks = class_tasks.reset_index(drop=True)
        self.n_datasets = len(class_tasks)
        return class_tasks
