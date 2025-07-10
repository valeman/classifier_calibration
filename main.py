 
import components.config as cf 
cf.logger.info("Importing resources")
import components.data as data
import components.eval_strat as e_s
import components.architectures as archs
import components.performance as perf
import os 

study_version = "v.1"
output_dir = "results"

if __name__ == "__main__":
    cf.logger.info("Starting experiment")
    
    ds_suite = data.DatasetSuite(study_version)
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)

    results = {}
    for ds in ds_suite:
        cf.logger.info(f"Dataset:{ds.df_name}")

        ds.convert_to_pandas()
        ds.pre_process("convert_unknown_to_nan") #Replace all "unknown" with nan
        
        ds.pre_process("detect_categorical") #Tag object columns as categorical
        ds.pre_process("convert_nan_to_'NON'") #Replace nan in cat columns with "non"
        ds.pre_process("encode_categoricals") 
        
        ds.pre_process("clean_numerical") #Ensure non-cat object columns only contain numbers.
        

        for architecture in architectures:
            cf.logger.info(f"Architecture:{architecture.name}")
            if architecture.name not in results.keys():
                results[architecture.name] = {}

            cf.logger.info(f"Start evaluation")
            eval = e_s.EvaluationStrategy(study_version, ds, architecture,  p_measures)  
            results[architecture.name][ds.df_name] = eval.run()
        break

    analysis = perf.AnalyzePerformance(study_version, results)
    #analysis.run()
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)
    analysis.save_to_disk(output_dir)