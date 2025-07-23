# classifier_calibration

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation
1. Clone the repository
2. Install python and docker 

## Usage
To run the project, use the following command:
```
python run.py

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