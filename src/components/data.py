from diskcache import Cache
import components.utils as util
import pandas as pd
import numpy as np
import tempfile
import openml
import os

base_tmp = tempfile.gettempdir() 
cache_dir = os.path.join(base_tmp, "download_cache")  
cache = Cache(directory=cache_dir)
cache_exp = 3600

@cache.memoize(expire=cache_exp)
def get_openml_task(taskid:int)-> tuple[pd.DataFrame,list[str]]:
    """
    Get the dataset of a taskid from openml and select metadata

    Args:
        taskid (int): The task id

    Returns:
        tuple[pd.DataFrame,list]: The dataset and which columns are categorical
    """
    try:
        task = openml.tasks.get_task(taskid)
        ds = task.get_dataset()
        df, _, cat_features, _ =  ds.get_data()
        cat_columns = [i for i,j in zip(df.columns, cat_features) if j]
        return df, cat_columns  
    except Exception:
        cache.evict((taskid, ))
        raise

@cache.memoize(expire=cache_exp)
def get_openml_study(studyid:int) -> pd.DataFrame:
    try:
        suite = openml.study.get_suite(studyid) 
        tasks = openml.tasks.list_tasks(task_id=suite.tasks, output_format="dataframe")
        return tasks
    except Exception:
        cache.evict((studyid, ))
        raise

def detect_categorical_columns(df:pd.DataFrame, sample_size:int = 100, fail_threshold:float = 0.5, random_seed:int=123) -> list:
    """
    Detect object columns that are likely categorical by sampling values
    and attempting to convert to float.

    Args:
        df (pd.DataFrame)
        sample_size (int, optional):  maximum number of non-null values to sample per column. Defaults to 100.
        fail_threshold (float, optional): proportion of failed conversions above which
                      the column is considered categorical. Defaults to 0.5.
        random_seed (int): A random seed
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
        sample = vals.sample(n=min(len(vals), sample_size), random_state=random_seed)
        
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
        df_encoded[col] = pd.Categorical(df_encoded[col].map(m), categories=list(m.values()))
        m = {str(k):str(v) for k,v in m.items()}
        mappings[col] = m
        
        
    return df_encoded, mappings


class Dataset:
    """
    The Dataset class holds a tabular dataset (df) and corresponding meta data.
    Also includes select functionality on dataframes. 
    """

    def __init__(self, df:None, df_name:str, meta_data:dict, random_seed:int=123):
        """
        Args:
            df (None): Any data structure representing a tabular dataset such as a pandas dataframe.
            df_name (str): The name of the dataset.
            meta_data (dict): All meta data such as Categorical columns, The target column etc. .
            random_seed (int=): A random seed.
        """
        self.df = df
        self.df_name = df_name
        self.random_seed = random_seed
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
                detected_cat_cols = detect_categorical_columns(self.df[self.non_cat_columns], random_seed=self.random_seed)
                self._update_categorial_meta_data(detected_cat_cols)
                
            case "encode_categoricals":
                cat_cols = self.meta_data["cat_columns"]
                self.df, e_map = encode_categoricals(self.df, cat_cols)
                self.meta_data["encoding_map"] = e_map
            
            case "convert_unknown_to_nan":
                self.df.replace(r"(?i)^unknown$", np.nan, regex=True, inplace=True)

            case "convert_nan_to_unique_val": #only for cat columns
                include_columns = self.meta_data["cat_columns"]
                
                for col in include_columns:
                    uniques = self.df[col].dropna().unique()
                    n_unique = util.get_unique_id(uniques, pre_fix="nan_", random_seed=self.random_seed)
                    
                    if pd.api.types.is_categorical_dtype(self.df[col]):
                    # Add n_unique to the category if it's not already there
                        if n_unique not in self.df[col].cat.categories:
                            self.df[col] = self.df[col].cat.add_categories([n_unique])
                    #Convert nan to new unique_str_val
                    self.df[col] = self.df[col].fillna(n_unique)
            
            case "clean_numerical": #Only for numeric columns with object dtype
                # a) Ensure we’re working with str (e.g. to strip whitespace)
                # b) Attempt to parse everything as float, unparseable → -1
                # c) Downcast to the smallest float dtype (float32 or Int64 if possible) 
                        
                for col in self.df[self.non_cat_columns].select_dtypes(include='object').columns:
                    num = pd.to_numeric(
                        self.df[col].astype(str).str.strip()
                        ,errors='coerce'
                        ,downcast='float'
                    )
                    
                    all_int = num.dropna().apply(float.is_integer).all()
                    if all_int:
                        # Cast to pandas nullable Int64
                        self.df[col] = num.astype('Int64')
                    else:
                        self.df[col] = num

            case "convert_nan_to_0": #Only for numerical columns
                self.df[self.non_cat_columns] = self.df[self.non_cat_columns].fillna(0) 

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

    def __init__(self, suite_name:str, random_seed:int=123):
        """
        Args:
            suite_name (str): The name of the dataset suite
            random_seed (int): A random seed.
        """
        self.suite_name = suite_name
        self.dataset_suite = None
        self.n_datasets = None
        self.random_seed=random_seed

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
                        ,random_seed=self.random_seed
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
        tasks = get_openml_study(studyid=457) #Study: TabArena-v0.1 Suite
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

               
