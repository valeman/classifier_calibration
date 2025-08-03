import platform
import os
import subprocess

IMAGE_NAME = "classifier-calibration:v1"
OUTPUT_DIR = os.path.join(os.getcwd(), "results")

# Ensure host output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build the image
subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)

# Determine the volume mount string
if platform.system() == "Windows":
    host_path = OUTPUT_DIR.replace("\\", "/")
else:
    host_path = OUTPUT_DIR

volume_arg = f"{host_path}:/app/results"

# Run the container
subprocess.run([
    "docker", "run",
    "--rm",
    "-v", volume_arg,
    IMAGE_NAME
], check=True)
