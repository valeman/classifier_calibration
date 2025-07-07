import pandas as pd
import openml


class Dataset:
    """
    The Dataset class holds a tabular dataset (df) and corresponding meta data.
    Also includes select functionality on dataframes. 
    """

    def __init__(self, df:None, df_name:str, source:str, target:str, cat_columns:list, meta_data:None):
        """
        Args:
            df (None): Any data structure representing a tabular dataset such as a pandas dataframe.
            df_name (str): The name of the dataset.
            source (str): The origin of df.
            target (str): The target column.
            cat_columns (list): Categorical columns.
            meta_data (None): Dump any residual meta data.
        """
        self.df = df
        self.df_name = df_name
        self.source = source
        self.target = target
        self.cat_columns = cat_columns
        self.meta_data = meta_data
        
    def to_pandas(self) -> pd.DataFrame:
        """
        Returns self.df as a pandas dataframe.

        Returns:
            pd.DataFrame
        """
        if isinstance(self.df, pd.DataFrame):
            return self.df
        
        else:
            raise NotImplementedError
        
        

class DatasetSuite:
    """
    The DatasetSuite class retrieves a collection of datasets (a suite) and returns them iteratively. 
    """

    def __init__(self, suite_name:str):
        """
        Args:
            suite_name (str): The name of the dataset suite
        """
        self.suite_name = suite_name
        self.dataset_suite = None
        self.n_datasets = None

        match suite_name:
            case "Tabarena-v0.1":
                self.dataset_suite = self.get_tabarena_v01()
            case "Tabarena-v0.1-Binary":
                self.dataset_suite = self.get_tabarena_v01_Binary()

            case _:
                raise NotImplementedError


    def __iter__(self) -> Dataset:
        """
        Iteratively collect each dataset in the suite from it's source into RAM. 
        Yield the corresponding dataset.
        
        Yields:
            Dataset
        """
        for i in range(self.n_datasets):
            match self.suite_name:
                case "Tabarena-v0.1" | "Tabarena-v0.1-Binary":
                    task_info = self.dataset_suite.iloc[i]
                    tid = int(task_info["tid"])
                    name = task_info["name"]
                    target = task_info["target_feature"]
                    df, cat_columns = get_openml_task(tid)
                    dataset = Dataset(df = df
                                    ,df_name = name
                                    ,source = "openml"
                                    ,target = target
                                    ,cat_columns = cat_columns
                                    ,meta_data = None
                    )
                case _:
                    raise NotImplementedError
            yield dataset

    def get_tabarena_v01(self):
        """
        Collects the classification tasks (datasets) of the TabArena-v0.1 Suite
        """
        suite = openml.study.get_suite(457) #Study: TabArena-v0.1 Suite
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        class_tasks = tasks[tasks["task_type"] == "Supervised Classification"]
        class_tasks = class_tasks[["tid", "name","NumberOfClasses", "target_feature"]]
        class_tasks = class_tasks.reset_index(drop=True)
        self.n_datasets = len(class_tasks)
        return class_tasks
    
    def get_tabarena_v01_Binary(self):
        """
        Collects the binary classification tasks (datasets) of the TabArena-v0.1 Suite
        """
        tasks = self.get_tabarena_v01()
        class_tasks = tasks[tasks["NumberOfClasses"] == 2]
        return class_tasks        


def get_openml_task(taskid):
    task = openml.tasks.get_task(taskid)
    ds = task.get_dataset()
    df, _, cat_features, at_names =  ds.get_data()
    cat_columns = [i for i,j in zip(df.columns, cat_features) if j]
    return df, cat_columns                 
