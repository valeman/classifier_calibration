import pandas as pd
import openml


def get_openml_task(taskid:int)-> tuple[pd.DataFrame,list]:
    """
    Get the dataset of a taskid from openml and select metadata

    Args:
        taskid (int): The task id

    Returns:
        tuple[pd.DataFrame,list]: The dataset and which columns are categorical
    """
    task = openml.tasks.get_task(taskid)
    ds = task.get_dataset()
    df, _, cat_features, at_names =  ds.get_data()
    cat_columns = [i for i,j in zip(df.columns, cat_features) if j]
    return df, cat_columns  


class Dataset:
    """
    The Dataset class holds a tabular dataset (df) and corresponding meta data.
    Also includes select functionality on dataframes. 
    """

    def __init__(self, df:None, df_name:str, meta_data:dict):
        """
        Args:
            df (None): Any data structure representing a tabular dataset such as a pandas dataframe.
            df_name (str): The name of the dataset.
            meta_data (dict): All meta data such as Categorical columns, The target column etc. .
        """
        self.df = df
        self.df_name = df_name
        self.meta_data = meta_data
        
    def pre_process(method:str) -> None:
        """
        Apply a transformation on self.df. 
        Should only support trivial transformations such as encodig text columns into numeric.
        Pre processing in general should lie under architectures.
        Args:
            method (str): How to pre process/transform self.df

        """
        match method:
            case "to_numeric":
                pass
            case _:
                raise NotImplementedError

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
            case "v.1":
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
                case "Tabarena-v0.1" | "v.1" :
                    task_info = self.dataset_suite.iloc[i]
                    tid = int(task_info["tid"])
                    name = task_info["name"]
                    target = task_info["target_feature"]
                    df, cat_columns = get_openml_task(tid)
                    cat_columns.remove(target)
                    dataset = Dataset(df = df
                                    ,df_name = name
                                    ,meta_data = {"source":"openml"
                                                  ,"target":target
                                                  ,"cat_features":cat_columns
                                                }
                    )
                case _:
                    raise NotImplementedError
            yield dataset

    def get_tabarena_v01(self) -> pd.DataFrame:
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
    
    def get_tabarena_v01_Binary(self) -> pd.DataFrame:
        """
        Collects the binary classification tasks (datasets) of the TabArena-v0.1 Suite
        """
        tasks = self.get_tabarena_v01()
        class_tasks = tasks[tasks["NumberOfClasses"] == 2]
        return class_tasks        

               
