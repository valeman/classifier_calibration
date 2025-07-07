import os 
import components.data as data 


if __name__ == "__main__":
    suite = data.DatasetSuite("Tabarena-v0.1")
    
    for ds in suite:
        break


        
    output_dir = "/results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "test.txt")
    with open(filename, "w") as f:
        f.write("FOOBAR")
    print(f"Wrote file: {filename}")