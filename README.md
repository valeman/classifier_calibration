# Classifier Calibration Evaluation Framework

This framework evaluates supervised tabular machine learning models on real binary classification problems, specifically analyzing performance changes after applying post-hoc calibration methods trained on a held-out calibration set. The evaluation uses the [TabArena-v0.1 Suite](https://www.openml.org/search?type=study&study_type=task&id=457) of datasets.

## Key Features
- **Models**: 15+  classifiers
- **Calibration Methods**: 5 post-hoc calibration methods
- **Robust Execution**: Containerized environment with resource monitoring
- **Reproducible Results**

## Models Evaluated
| Model | Implementation |
|-------|----------------|
| Support Vector Machine | [scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) |
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
- **System**: Linux with cgroupv2 (WSL2 supported on Windows)
- **Software**:
  - Python 3.10+
  - Docker 20.10+
  - systemd (for service management)

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/valeman/classifier_calibration.git
cd classifier_calibration
```
### 1. Verify System Requirements
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
Results will be saved to the ./results/ directory including:

- Performance measures per architecture, fold, and dataset.
- A plethora of plots summarizing findings. 


## Manual Execution (Without Docker)
### 1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### 2. Install dependencies
### 3. Run python files
```bash
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/main.py > "job.log" 2>&1'
```
```bash
systemd-run --user --unit=job-1 --quiet --no-block \
  bash -c 'cd YOUR_PATH/classifier_calibration && source venv/bin/activate && python src/analyse_results.py >> "job.log" 2>&1'
```


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






