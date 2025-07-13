 
import components.config as cf 
cf.logger.info("Importing resources")
import components.data as data
import components.eval_strat as e_s
import components.architectures as archs
import components.performance as perf
import traceback
import os 


study_version = "v.1"
output_dir = "results"

if __name__ == "__main__":

    cf.logger.info("Defining experiment")
    ds_suite = data.DatasetSuite(study_version)
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)
    results = {}

    cf.logger.info("Starting experiment")
    for ds in ds_suite:
        cf.logger.info(f"Dataset:{ds.df_name}")
        cf.logger.info(f"Start common pre-processing")
        ds.convert_to_pandas()
        ds.pre_process("convert_unknown_to_nan") #Replace all "unknown" with nan
        ds.pre_process("detect_categorical") #Tag object columns as categorical
        ds.pre_process("convert_nan_to_'NON'") #Replace nan in cat columns with "non"
        ds.pre_process("encode_categoricals") 
        ds.pre_process("clean_numerical") #Ensure non-cat object columns only contain numbers.
        cf.logger.info(f"End common pre-processing")
        
        for architecture in architectures:
            
            cf.logger.info(f"Evaluate Architecture:{architecture.name}")
            if architecture.name not in results.keys():
                results[architecture.name] = {}
            
            
            cf.logger.info(f"Run evaluation")
            eval = e_s.EvaluationStrategy(study_version, ds, architecture,  p_measures)  
            try:
                results[architecture.name][ds.df_name] = eval.run()

            except Exception as err:
                cf.logger.exception(f"Failed to evaluate {architecture.name} on {ds.df_name}")
                results[architecture.name][ds.df_name] = {
                    "status": "failed"
                    ,"error_message": str(err)
                    ,"trace": traceback.format_exc()
                }
            cf.logger.info(f"End evaluation")

        break
    cf.logger.info("Experiment completed")
        
    cf.logger.info("Analyse performance")
    analysis = perf.AnalyzePerformance(study_version, results)
    #analysis.run()
    
    cf.logger.info("Export results")
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    analysis.save_to_disk(output_dir)
    