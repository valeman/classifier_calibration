# classifier_calibration
Classifier calibration v.1 contains python scripts which evalutes tabular machine learning models on real supervised binary classification problems. Specifically we evaluate the performance of select models and thereafter changes in performance after applying select post-hoc calibration methods trained on a held-out calibration set, over a dataset suite. 

The dataset suite used is : [TabArena-v0.1 Suite](https://www.openml.org/search?type=study&study_type=task&id=457)

The models examined are:<br>
    [Support vector machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)<br>
    [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>
    [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)<br>
    [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)<br>
    [ExtraTrees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)<br>
    [Explainable Boosting Machine](https://interpret.ml/docs/ebm.html)<br>
    [Catboost](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)<br>
    [XGBoost](https://federated-xgboost.readthedocs.io/en/latest/python/python_api.html)<br>
    [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)<br>
    [ModernNCA](https://github.com/autogluon/tabrepo)<br>
    [TabTransformer](https://pytorch-tabular.readthedocs.io/en/latest/)<br>
    [TabICL](https://github.com/autogluon/tabrepo)<br>
    [TabPFN](https://github.com/PriorLabs/TabPFN)<br>
    [TabM](https://github.com/autogluon/tabrepo)<br>
    [Multilayer Perceptron](https://sklearner.com/scikit-learn-mlpclassifier/)<br> 
    [Real Multilayer Perceptron](https://github.com/autogluon/tabrepo)<br>

The post-hoc calibration methods examined are:<br>
    [Platt scaling](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)<br>
    [Isotonic regression](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)<br>
    [Beta calibration](https://github.com/betacal/python)<br>
    [Venn-abers](https://github.com/ip200/venn-abers)<br>
    [Pearsonify](https://github.com/xRiskLab/pearsonify)<br>

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
cd YOUR_PATH/classifier_calibration
```
Ensure you have docker, python and cgroupv2:
```
python -V 
docker -v
stat -fc %T /sys/fs/cgroup
```
Run the shell script run.sh: 
```
./run.sh
```

If you don't want to run the project as a docker image replicate the enviornment defined in the Dockerfile.
Thereafter run:
```
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/main.py > "out.log" 2>&1'
```
```
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/analyse_results.py > "out.log" 2>&1'
```

## Tips
Start the docker daemon
```
sudo systemctl start docker
```
Test and see your cgroupv2 works as expected. 
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
systemctl --user reset-failed
journalctl --user -u job-1.service
```
Monitor resource usage 
```
ps -eo user,%cpu,%mem --sort=user | awk 'NR==1{print;next} {cpu[$1]+=$2; mem[$1]+=$3} END {for (u in cpu) printf "%-15s %6.2f%% CPU  %6.2f%% MEM\n", u, cpu[u], mem[u]}'
htop
```







