from tqdm import tqdm
import components.config as cf 
import components.data as data
import components.eval_strat as es
import components.architectures as archs
import components.performance as perf
import traceback
import os 


study_version = "v.1"
ds_suite_name = "Tabarena-v0.1-binary"
eval_strat_name = "nested-5-fold-CV"
output_dir = "results"

if __name__ == "__main__":

    cf.logger.info("Defining experiment")
    datasets = data.DatasetSuite(ds_suite_name)
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)
    results = {}

    cf.logger.info("Starting experiment")
    for ds in tqdm(datasets, desc=f"Dataset suite", unit="ds", total=datasets.n_datasets):
        cf.logger.info(f"Dataset name:{ds.df_name}")

        cf.logger.info(f"Start common pre-processing")
        ds.convert_to_pandas()
        ds.pre_process("convert_unknown_to_nan") #Replace all "unknown" with nan
        ds.pre_process("detect_categorical") #Tag object columns as categorical
        ds.pre_process("convert_nan_to_'NON'") #Replace nan in cat columns with "non"
        ds.pre_process("encode_categoricals") 
        ds.pre_process("clean_numerical") #Ensure non-cat object columns only contain numbers.
        
        cf.logger.info(f"End common pre-processing")
        
        for arch in tqdm(architectures, desc=f"Architecture suite", unit="arch", total=architectures.n_architectures):
            cf.logger.info(f"Evaluate Architecture:{arch.name}")
            if arch.name not in results.keys():
                results[arch.name] = {}
            
            
            cf.logger.info(f"Run evaluation")
            eval_strat = es.EvaluationStrategy(eval_strat_name, ds, arch,  p_measures)  
            try:
                results[arch.name][ds.df_name] = eval_strat.run()

            except Exception as err:
                cf.logger.exception(f"Failed to evaluate {arch.name} on {ds.df_name}")
                results[arch.name][ds.df_name] = {
                    "status": "failed"
                    ,"error_message": str(err)
                    ,"trace": traceback.format_exc()
                }
            cf.logger.info(f"End evaluation")

    cf.logger.info("Experiment completed")
        
    cf.logger.info("Analyse performance")
    analysis = perf.AnalyzePerformance(study_version, results)
    #analysis.run()
    
    cf.logger.info("Export results")
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    analysis.save_to_disk(output_dir)
    