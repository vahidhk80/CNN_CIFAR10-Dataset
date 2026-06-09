# PyTorch Training Workflow with Experiment Tracking and Distributed Execution

## Project Overview

This project demonstrates a professional PyTorch machine learning workflow developed for ICTPRG435 Write Scripts for Software Applications and ICTCLD401 Configure Cloud Services. The project includes model training, evaluation, experiment tracking, checkpointing, reproducibility controls, and distributed execution using a Bash launcher.

The solution was developed using Kaggle GPU resources and accessed remotely through Visual Studio Code using a VSCode Tunnel connection.

---

## Environment Setup

### Development Environment

* Kaggle Notebook with GPU enabled
* Visual Studio Code connected through VSCode Tunnel
* GitHub repository for version control
* Python 3.12
* NVIDIA Tesla T4 GPU (Kaggle)

### Clone Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---
## Repository Structure

```text
.
├── train.py
├── run_distributed.sh
├── requirements.txt
├── README.md
├── evidence/
│   ├── part_a/
│   ├── part_b/
│   └── part_c/
└── runs/
```

---

## Running the Training Script

Example:

```bash
python train.py --epochs 5 --batch-size 128 --lr 0.001 --tracker wandb
```

Available arguments:

```text
--epochs
--batch-size
--lr
--tracker {wandb, mlflow, none}
```

---

## Running Distributed Training

Example:

```bash
bash run_distributed.sh --epochs 5 --batch-size 128 --lr 0.001 --tracker wandb
```

The launcher automatically:

* Detects available GPUs
* Launches training using torchrun when multiple GPUs are detected
* Falls back to single-process execution when required
* Creates timestamped logs

---

## Experiment Tracking

### Weights & Biases

The training script supports experiment tracking using Weights & Biases.

Tracked information:

* Training loss
* Accuracy
* Hyperparameters
* Model checkpoints

### MLflow

The script also supports MLflow tracking and artifact storage.

---

## Reproducibility

The project includes:

* Fixed random seeds
* Requirements file with package versions
* Deterministic configuration where applicable
* Saved checkpoints for model recovery

---

## Generated Artifacts

The project generates:

* Model checkpoints (.pth)
* Training logs
* Experiment tracking records
* Evaluation metrics
* Confusion matrix outputs

---

## Troubleshooting

### CUDA Not Available

Check GPU availability:

```bash
nvidia-smi
```

### Missing Dependencies

Install project requirements:

```bash
pip install -r requirements.txt
```

### WandB Login Error

Login before training:

```bash
wandb login
```

### MLflow Database Error

Ensure the tracking directory exists and is writable.

---

## Evidence Folder Structure

```text
evidence/
├── part_a/
├── part_b/
└── part_c/
```

Evidence includes:

* Environment setup screenshots
* Training logs
* Experiment tracking screenshots
* Model checkpoints
* Distributed launcher outputs

---

## Mapping to Unit of Competency Criteria

### ICTPRG435 Write Scripts for Software Applications

* Data loading and preprocessing
* Model implementation
* Training loop development
* Experiment tracking
* Error handling
* Command-line interface development
* Distributed execution scripting

### ICTCLD401 Configure Cloud Services

* Kaggle cloud environment configuration
* GPU resource utilisation
* Remote VSCode tunnel connection
* Dependency management
* Cloud-based model execution
* Environment documentation and setup

```
```
