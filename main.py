 
import components.data as data
import components.eval_strat as e_s
import components.architectures as archs
import components.performance as perf
import os 

study_version = "v1" 
if __name__ == "__main__":
    
    suite = data.DatasetSuite("Tabarena-v0.1-Binary")
    p_measures = perf.PerformanceMeasures(study_version)
    architectures = archs.ArchitectureSuite(study_version)
    results = {}

    for ds in suite:
        for architecture in architectures:
            eval = e_s.EvaluationStrategy("nested-5-fold-CV", ds, architecture,  p_measures)
            results[architecture.name] = eval.run()
        
    
    ### Analyze and do things with the result
    ### Save results to disk 
    # output_dir = "/results"
    # os.makedirs(output_dir, exist_ok=True)
    # filename = os.path.join(output_dir, "test.txt")
    # with open(filename, "w") as f:
    #     f.write("FOOBAR")
    # print(f"Wrote file: {filename}")