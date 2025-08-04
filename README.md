# classifier_calibration

## Table of Contents
- [Requirements](#Requirements)
- [Usage](#Usage)
- [Tips](#Tips)

## Requirements
1. Clone the repository
2. Install python and docker 
3. A Linux host machine with cgroupv2

## Usage
To run the project, use the following commands:

Ensure the repository's root directory is your current working directory. 
```
cd your_path/classifier_calibration
```
Ensure you have docker, python and cgroupv2:
```
python -V 
docker -v
stat -fc %T /sys/fs/cgroup
```
Run the python file run.py as a service: 
```
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'python run.py > "out.log" 2>&1'
```

If you don't want to run the project as a docker image replicate the enviornment defined in the Dockerfile.
Thereafter run:
```
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd your_path/classifier_calibration && source venv/bin/activate && python src/main.py > "out.log" 2>&1'
```
```
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd your_path/classifier_calibration && source venv/bin/activate && python src/analyse_results.py > "out.log" 2>&1'
```

## Tips
Test and see you're cgroupv2 works as expected. 
```
systemd-run --user --scope --unit=my-python-job python src/components/resource_tracker.py
```
Enable lingering
```
loginctl enable-linger $(id -un)
```
Manage the serivce
```
systemctl --user list-units --type=service
systemctl --user stop job-1.service
journalctl --user -u job-1.service
```
Monitor resource usage 
```
ps -eo user,%cpu,%mem --sort=user | awk 'NR==1{print;next} {cpu[$1]+=$2; mem[$1]+=$3} END {for (u in cpu) printf "%-15s %6.2f%% CPU  %6.2f%% MEM\n", u, cpu[u], mem[u]}'
htop
```







