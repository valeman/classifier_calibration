# Classifier Calibration Evaluation Framework

This framework evaluates supervised tabular machine learning models on real binary classification problems, specifically analyzing performance changes after applying post-hoc calibration methods trained on a held-out calibration set. The evaluation uses the [TabArena-v0.1 Suite](https://www.openml.org/search?type=study&study_type=task&id=457) of datasets.

## Key Features
- **Models**: 15+  classifiers
- **Calibration Methods**: 5 post-hoc calibration methods
- **Robust Execution**: Containerized environment with resource monitoring
- **Reproducability** 


## Project layout (important files)

```
./run.sh                # systemd-based launch helper (user service)
./run.py                # builds Docker image and runs container
./Dockerfile            # container environment (python:3.12-slim)
./src/main.py           # experiment runner (evaluates architectures)
./src/analyse_results.py# analyser / plotting pipeline
./requirements.txt      # python deps (used in Dockerfile)
./results/              # output produced by run (mounted volume)
```

## Reproducability

`main.py` intentionally seeds multiple RNGs to reduce nondeterminism:

- `SEED = 123456789` — applied to `random.seed()`, `numpy.random.seed()`, `torch.manual_seed()` and `PYTHONHASHSEED`.
- Threading / BLAS environments are pinned with environment variables and PyTorch thread controls:
  - `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `KMP_AFFINITY` and PyTorch `set_num_threads` / `set_num_interop_threads`.

**Note:** perfect bitwise reproducibility across different kernel versions, CPU architectures, BLAS libraries, or PyTorch builds is *not guaranteed*. The seed and thread controls make runs highly consistent for research comparisons but small numeric drift is still possible. 


## Models Evaluated
| Model | Implementation |
|-------|----------------|
| Empirical class distribution of target (Dummy) | [scikit-learn DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier) |
| Support Vector Machine | [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
| Linear Discriminant Analysis | [scikit-learn LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) |
| Naïve bayes | [scikit-learn GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB) |
| Gaussian process classification | [scikit-learn GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html#sklearn.gaussian_process.GaussianProcessClassifier) |
| Logistic Regression | [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
| K-Nearest Neighbours | [scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) |
| Random Forest | [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) |
| ExtraTrees | [scikit-learn ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) |
| Explainable Boosting Machine | [InterpretML EBM](https://interpret.ml/docs/ebm.html) |
| CatBoost | [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier) |
| XGBoost | [XGBoost Python API](https://federated-xgboost.readthedocs.io/en/latest/python/python_api.html) |
| LightGBM | [LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html) |
| ModernNCA | [TabRepo](https://github.com/autogluon/tabrepo) |
| TabTransformer | [PyTorch Tabular](https://pytorch-tabular.readthedocs.io/en/latest/) |
| TabICL | [TabRepo](https://github.com/autogluon/tabrepo) |
| TabPFN | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) |
| TabM | [TabRepo](https://github.com/autogluon/tabrepo) |
| Multilayer Perceptron | [scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |
| Real MLP | [TabRepo](https://github.com/autogluon/tabrepo) |

## Calibration Methods
| Method | Implementation |
|--------|----------------|
| Platt Scaling | [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) |
| Isotonic Regression | [scikit-learn IsotonicRegression](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html) |
| Beta Calibration | [betacal](https://github.com/betacal/python) |
| Venn-Abers | [venn-abers](https://github.com/ip200/venn-abers) |
| Pearsonify | [pearsonify](https://github.com/xRiskLab/pearsonify) |

## Requirements
- **System**: 
  - Ubuntu linux (or equivalent) with cgroupv2 mounted at `/sys/fs/cgroup` (WSL2 supported on Windows)
  - `systemd` available for user systemd services
- **Software**:
  - Python 3.10+
  - Docker 20.10+ and the user is in the `docker` group
  - systemd 245+ (for service management)

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/valeman/classifier_calibration.git
cd classifier_calibration
```
### 2. Verify System Requirements
```bash
# Check Python version
python3 --version

# Check Docker version
docker --version

# Verify cgroupv2
stat -fc %T /sys/fs/cgroup  # Should return 'cgroup2fs'
```
### 3. Configure System Permissions
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # Apply group changes without logout

# Enable lingering for systemd services
loginctl enable-linger $USER

# Start Docker service
sudo systemctl start docker
```
### 4. Run the Evaluation

```bash
# Execute the main pipeline (builds Docker image and runs evaluation)
./run.sh

# Monitor progress
tail -f launch.log  # System service logs
tail -f job.log     # Job execution logs
```
### 5. Review results
After a successful run the `./results/` directory should contain:

```
.src/results/
├─ results.txt           # Nested dict with measurements (arch -> dataset -> [runs])
├─ datasets_md.txt       # Per-dataset metadata collected by DatasetSuite
├─ experiment_md.txt     # Experiment metadata (seed, machine specs)
├─ assets/               # Plots and images generated by analyse_results.py
└─ <optional-subdirs>    # If merging outputs across jobs/runs
```


## Manual Execution (Without Docker)
### 1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 2. Replicate the Dockerfile environment 
### 3. Run python files:
```bash
systemd-run --user --unit=job-1 --quiet --no-block --property=Delegate=yes \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/main.py > "job.log" 2>&1'
```
```bash
systemd-run --user --unit=job-1 --quiet --no-block --property=Delegate=yes \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/analyse_results.py >> "job.log" 2>&1'
```
## where to start debugging:

- `launch.log` — environment dump, `systemctl --user` status, last journal entries before launch and service status just after launch.
- `job.log` — the stdout/stderr of the process (the Python program inside the service). This includes logger output from `main.py` and `analyse_results.py`.

When something fails, start with `launch.log` to see if systemd/docker/env issues occurred, then inspect `job.log` for the docker/Python logs.


## Monitoring and Management
### Service Management 
```bash
# List active services
systemctl --user list-units --type=service

# Check service status
systemctl --user status job-1

# Stop service
systemctl --user stop job-1.service

# Reset failed services
systemctl --user reset-failed

# View service logs
journalctl --user -u job-1.service

# View last 100 log entries
journalctl --user -u job-1.service -n 100

# Follow logs in real-time
journalctl --user -u job-1.service -f
```
### Resource Monitoring
```bash
# System resource usage
htop

# Per-user resource consumption
ps -eo user,%cpu,%mem --sort=user | \
  awk 'NR==1{print;next} {cpu[$1]+=$2; mem[$1]+=$3} END \
  {for (u in cpu) printf "%-15s %6.2f%% CPU  %6.2f%% MEM\n", u, cpu[u], mem[u]}'

# Docker container stats
docker stats
```

## License
This project is licensed under the [MIT License](LICENSE).






