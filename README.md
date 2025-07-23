# classifier_calibration

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation
1. Clone the repository
2. Install python and docker 

## Usage
To run the project, use the following commands:

Ensure the repository's root directory is your current working directory. 
```
cd your_path/classifier_calibration
```
Ensure your python version is 3.12.*
```
python -V 
```
Ensure you have virtualenv installed
```
pip install virtualenv
```
Create the virtualenv 
```
python -m venv venv
```
Activate the virtual env.
If on windows:
```
cd venv/Scripts
activate
cd ../..
```
If on Linux/MacOs:
```
source venv/bin/activate
```
Install all the requirements
```
pip install -r requirements.txt
```
Run the main.py script to evaluate all the models over all datasets
```
python main.py
```
Run the analyse_results.py script to produce all plots.
All plots are exported to /archive/assets/*
```
python analyse_results.py
```



## For later

```
nohup python main.py > out_3_4.log 2>&1 &

ps -eo user,%cpu,%mem --sort=user | awk 'NR==1{print;next} {cpu[$1]+=$2; mem[$1]+=$3} END {for (u in cpu) printf "%-15s %6.2f%% CPU  %6.2f%% MEM\n", u, cpu[u], mem[u]}'
htop

# 1. Look for the cgroup2 filesystem
$ grep -E 'cgroup2' /proc/filesystems
nodev   cgroup2

# 2. Confirm it’s actually mounted
$ findmnt -t cgroup2 /sys/fs/cgroup
TARGET        SOURCE    FSTYPE OPTIONS
/sys/fs/cgroup none      cgroup2 rw,nosuid,nodev,noexec,relatime

# 3. Or check directly:
$ stat -fc %T /sys/fs/cgroup
cgroup2fs


```