 
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
    suite = data.DatasetSuite(study_version)
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)
    #results = {}

    for ds in suite:
        break
        #for architecture in architectures:
        #    eval = e_s.EvaluationStrategy(study_version, ds, architecture,  p_measures)
        #    results[architecture.name] = eval.run()
        
    
    ### Analyze and do things with the result
    ### Save results to disk 
    # output_dir = "results"
    # output_dir = os.path.join(os.getcwd(), output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    # filename = os.path.join(output_dir, "test.txt")
    # with open(filename, "w") as f:
    #     f.write("FOOBAR")
    # cf.logger.info(f"Result written too: {filename}")'