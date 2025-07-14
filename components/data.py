import pandas as pd
import numpy as np
import openml
import components.config as cf

SEED = cf.SEED

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
    df, _, cat_features, _ =  ds.get_data()
    cat_columns = [i for i,j in zip(df.columns, cat_features) if j]
    return df, cat_columns  


def detect_categorical_columns(df:pd.DataFrame, sample_size:int = 100, fail_threshold:float = 0.5) -> list:
    """
    Detect object columns that are likely categorical by sampling values
    and attempting to convert to float.

    Args:
        df (pd.DataFrame)
        sample_size (int, optional):  maximum number of non-null values to sample per column. Defaults to 100.
        fail_threshold (float, optional): proportion of failed conversions above which
                      the column is considered categorical. Defaults to 0.5.

    Returns:
        list: List of column names likely to be categorical
    """

    cat_cols = []
    for col in df.select_dtypes(include='object').columns:
        # drop nulls and cast to str in case there are mixed types
        vals = df[col].dropna().astype(str).drop_duplicates()
        if vals.empty:
            continue
        
        # sample up to sample_size values
        sample = vals.sample(n=min(len(vals), sample_size), random_state=SEED)
        
        # try converting each to float
        fail_count = 0
        for v in sample:
            try:
                float(v)
            except ValueError:
                fail_count += 1
        
        # if majority fail, flag as categorical
        if fail_count / len(sample) > fail_threshold:
            cat_cols.append(col)

    return cat_cols

def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict]]:
    """ 
    Encode each column in cat_cols of df to positive integers.
    - if a column has exactly 2 uniques, maps to {v0: 0, v1: 1}
    - otherwise maps sorted unique values to 1,2,3,...

    Args:
        df (pd.DataFrame): The df to encode
        cat_cols (list[str]): A list of categorical columns to encode

    Returns:
        tuple[pd.DataFrame, dict[str, dict]]: : the encoded df and the mapping
    """
    df_encoded = df.copy()
    mappings = {}

    for col in cat_cols:
        uniques = sorted(df_encoded[col].dropna().unique())
        if len(uniques) == 2:
            codes = [0, 1]
        else:
            codes = list(range(1, len(uniques) + 1))

        m = {val: int(code) for val, code in zip(uniques, codes)}
        mappings[col] = m
        
        df_encoded[col] = pd.Categorical(df_encoded[col].map(m), categories=list(m.values()))
    return df_encoded, mappings


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
        self.meta_data["n_columns"] = len(self.df.columns)
        self.meta_data["n_rows"] = len(self.df)
    
    @property
    def non_cat_columns(self) -> list[str]:
        """
        Return all column names not flagged as categorical.

        Returns:
            list[str]
        """
        cc = self.meta_data["cat_columns"]
        nct = [c for c in self.df.columns if c not in cc]
        return nct       


    def pre_process(self,method:str) -> None:
        """
        Apply a transformation on self.df
        Should only support common transformations such as encodig text columns into numeric.
        Pre processing in general should be architecture specific.
        Args:
            method (str): How to pre process self.df

        """
        match method:
            case "detect_categorical":
                detected_cat_cols = detect_categorical_columns(self.df[self.non_cat_columns])
                self._update_categorial_meta_data(detected_cat_cols)
                
            case "encode_categoricals":
                cat_cols = self.meta_data["cat_columns"]
                self.df, e_map = encode_categoricals(self.df, cat_cols)
                self.meta_data["encoding_map"] = e_map
            
            case "convert_unknown_to_nan":
                self.df.replace(r"(?i)^unknown$", np.nan, regex=True, inplace=True)

            case "convert_nan_to_'NON'": #only for cat columns
                include_columns = self.meta_data["cat_columns"]

                for col in include_columns:
                    if pd.api.types.is_categorical_dtype(self.df[col]):
                    # Add NON to the category if it's not already there
                        if 'NON' not in self.df[col].cat.categories:
                            self.df[col] = self.df[col].cat.add_categories(['NON'])
 
                self.df[include_columns] = self.df[include_columns].fillna('NON')
            
            case "clean_numerical": #only for cat columns object columns
                # a) Ensure we’re working with str (e.g. to strip whitespace)
                # b) Attempt to parse everything as float, unparseable → NaN
                # c) Downcast to the smallest float dtype (float32 or Int64 if possible) 
                        
                for col in self.df[self.non_cat_columns].select_dtypes(include='object').columns:
                    self.df[col] = pd.to_numeric(
                        self.df[col].astype(str).str.strip()
                        ,errors='coerce'
                        ,downcast='float'
                    )
                    
                    if self.df[col].dropna().apply(float.is_integer).all():
                        # Cast to pandas nullable Int64
                        self.df[col] = self.df[col].astype('Int64')

            case _:
                raise NotImplementedError
            
    def _update_categorial_meta_data(self, detected_cat_cols:list[str]) -> None:
        """
        Updates meta data to reflect all categorical features. 

        Args:
            detected_cat_cols (list[str]): A list of detected categorical features.
        """
        self.meta_data["detected_cat_cols"] = detected_cat_cols
        self.meta_data["cat_columns"].extend(detected_cat_cols)
        self.meta_data["cat_features"].extend(detected_cat_cols)
        self.meta_data["non_cat_features"] =  self.non_cat_columns
        features = [c for c in self.df.columns if c != self.meta_data["target"]]
        cat_features_indices = [i for i,c in enumerate(self.df[features].columns) if c in self.meta_data["cat_features"]]   
        self.meta_data["cat_features_indices"] = cat_features_indices
        
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
        
    def convert_to_pandas(self) -> None:
        """
        Convert self.df to a pandas dataframe.
        """
        self.df = self.to_pandas().copy()    


class DatasetSuite:
    """
    The DatasetSuite class retrieves a collection of datasets (a suite) and returns each dataset iteratively. 
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
            case "Tabarena-v0.1-class":
                self.dataset_suite = self.get_tabarena_v01_class()
            case "Tabarena-v0.1-binary":
                self.dataset_suite = self.get_tabarena_v01_binary()
            case _:
                raise NotImplementedError
            
        self.n_datasets = len(self.dataset_suite)
        
    def __iter__(self) -> Dataset:
        """
        Iteratively collect each dataset in the suite from it's source into RAM. 
        Yield the corresponding dataset.
        
        Yields:
            Dataset
        """

        for i in range(self.n_datasets):
            match self.suite_name:
                case "Tabarena-v0.1-class" | "Tabarena-v0.1-binary":
                    dataset = self._load_openml_class_task(i)
                case _:
                    raise NotImplementedError
            yield dataset

    def _load_openml_class_task(self, index:int) -> Dataset:
        """
        Download the openml classification task at the given index in the suite.
        Return the dataset

        Args:
            index (int)
        
        Returns:
            Dataset
        """
        task_info = self.dataset_suite.iloc[index]
        
        tid = int(task_info["tid"])
        name = task_info["name"]
        target = task_info["target_feature"]
        
        df, cat_columns = get_openml_task(tid)
        
        if target not in cat_columns: 
            cat_columns.append(target)

        cat_features = cat_columns.copy()
        cat_features.remove(target)

        dataset = Dataset(df = df
                        ,df_name = name
                        ,meta_data = {"source":"openml"
                                        ,"target":target
                                        ,"cat_columns":cat_columns
                                        ,"cat_features":cat_features
                        }
        )
        return dataset

    def get_tabarena_v01_class(self) -> pd.DataFrame:
        """
        Collects the classification tasks (datasets) of the TabArena-v0.1 Suite
        """
        suite = openml.study.get_suite(457) #Study: TabArena-v0.1 Suite
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        class_tasks = tasks[tasks["task_type"] == "Supervised Classification"]
        class_tasks = class_tasks[["tid", "name","NumberOfClasses", "target_feature"]]
        class_tasks = class_tasks.reset_index(drop=True)
        return class_tasks
    
    def get_tabarena_v01_binary(self) -> pd.DataFrame:
        """
        Collects the binary classification tasks (datasets) of the TabArena-v0.1 Suite
        """
        class_tasks = self.get_tabarena_v01_class()
        binary_class_tasks = class_tasks[class_tasks["NumberOfClasses"] == 2]
        return binary_class_tasks        

               
