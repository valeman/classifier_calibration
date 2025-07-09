 
import components.config as cf 
cf.logger.info("Importing resources")
import components.data as data
import components.eval_strat as e_s
import components.architectures as archs
import components.performance as perf
import os 

study_version = "v.1"

if __name__ == "__main__":
    cf.logger.info("Starting experiment")
    
    ds_suite = data.DatasetSuite(study_version)
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)

    results = {}
    for ds in ds_suite:
        cf.logger.info(f"Dataset:{ds.df_name}")

        ds.convert_to_pandas()
        ds.pre_process("encode_categoricals")
        
        for architecture in architectures:
            cf.logger.info(f"Architecture:{architecture.name}")
            eval = e_s.EvaluationStrategy(study_version, ds, architecture,  p_measures)
            
            if architecture.name not in results.keys():
                results[architecture.name] = {}
            
            cf.logger.info(f"Start evaluation")
            results[architecture.name][ds.df_name] = eval.run()
        break

    analysis = perf.AnalyzePerformance(study_version, results)
    analysis.run()

    # output_dir = "results"
    # output_dir = os.path.join(os.getcwd(), output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    # filename = os.path.join(output_dir, "results.txt")
    # analysis.save_to_disk(filename)

    ### Analyze and do things with the result
    ### Save results to disk 
    # output_dir = "results"
    # output_dir = os.path.join(os.getcwd(), output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    # filename = os.path.join(output_dir, "test.txt")
    # with open(filename, "w") as f:
    #     f.write("FOOBAR")
    # cf.logger.info(f"Result written too: {filename}")'