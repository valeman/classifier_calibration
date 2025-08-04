import platform
import os
import subprocess

IMAGE_NAME = "classifier-calibration:v1"
OUTPUT_DIR = os.path.join(os.getcwd(), "results")

# Ensure the OS is not Windows
if platform.system() == "Windows":
    raise OSError("Host OS must be Linux with cgroupv2")

# Ensure host output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build the image
subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)

# Run the container
volume_arg = f"{OUTPUT_DIR}:/app/results"
subprocess.run([
    "docker", "run",
    "--rm",
    "-v", volume_arg,
    IMAGE_NAME
], check=True)
