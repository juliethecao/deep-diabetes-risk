# Deep Diabetes Risk

Diabetes risk prediction on BRFSS 2015 tabular health indicators using:

- Baselines: Logistic Regression, KNN, Gaussian Naive Bayes
- Final model: Deep MLP (PyTorch)

The repository includes preprocessing, training, evaluation, and plotting scripts so results can be reproduced from scratch.

## Project Overview

This project predicts binary diabetes status (`Diabetes_binary`) from survey-based health indicators.

- Input: 21 tabular health features
- Output: probability of diabetes (binary classification)
- Pipeline:
	1. Preprocess CSV data into PyTorch tensor splits
	2. Train Deep MLP with class-imbalance handling
	3. Evaluate Deep MLP and baseline models
	4. Save metrics tables and publication-ready plots

## Repository Structure

- [data](data): raw CSV files
- [data_processed](data_processed): processed tensor splits (`train.pt`, `val.pt`, `test.pt`)
- [src/data_pipeline.py](src/data_pipeline.py): preprocessing pipeline
- [src/deep_mlp.py](src/deep_mlp.py): Deep MLP training + test evaluation
- [src/baselines.py](src/baselines.py): baseline model definitions
- [src/plot_results.py](src/plot_results.py): baseline vs MLP evaluation + plots
- [src/plot_utils.py](src/plot_utils.py): plotting utilities
- [src/results](src/results): saved model, metrics, and figures
- [demo_notebook.ipynb](demo_notebook.ipynb): demo notebook for running the workflow

## Setup Instructions

### 1) Create environment and install dependencies

From project root:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Verify expected Python packages

Dependencies are listed in [requirements.txt](requirements.txt):

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch
- torchvision
- tqdm
- jupyter

## Dataset

### Where to download

The project uses the CDC BRFSS 2015 diabetes health indicators dataset.

- Kaggle source: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

### Dataset files used

The repository currently includes CSV files under [data](data), including:

- `diabetes_binary_health_indicators_BRFSS2015.csv` (used by preprocessing)
- `diabetes_012_health_indicators_BRFSS2015.csv`
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

### Small sample dataset requirement

The full dataset used is public (not private)

## How to Preprocess Data

Run from project root:

```bash
python src/data_pipeline.py
```

What this does:

- Loads `data/diabetes_binary_health_indicators_BRFSS2015.csv`
- Stratified split: 70% train, 15% val, 15% test
- Standardizes numeric columns (`BMI`, `GenHlth`, `MentHlth`, `PhysHlth`, `Age`, `Education`, `Income`)
- Saves tensor datasets to [data_processed](data_processed):
	- `train.pt`
	- `val.pt`
	- `test.pt`

## How to Train the Model

Run from project root:

```bash
python src/deep_mlp.py
```

Training script behavior:

- Uses weighted BCE loss for imbalance handling
- Optimizer: AdamW
- LR scheduler: CosineAnnealingLR
- Early stopping on validation ROC-AUC
- Saves best model to [src/results/best_model.pt](src/results/best_model.pt)
- Saves Deep MLP test metrics to [src/results/test_metrics.json](src/results/test_metrics.json)

## How to Evaluate the Model

Run from project root:

```bash
python src/plot_results.py
```

Evaluation script behavior:

- Trains/evaluates baselines on train/test splits
- Loads trained Deep MLP and compares against baselines
- Prints metrics table in terminal
- Saves metrics table to [src/results/model_metrics_table.txt](src/results/model_metrics_table.txt)
- Generates all comparison/evaluation plots

## Expected Outputs

After running preprocessing + training + evaluation, expect these files in [src/results](src/results):

- `best_model.pt`
- `test_metrics.json`
- `model_metrics_table.txt`
- `roc_all_models.png`
- `pr_all_models.png`
- `calibration_all_models.png`
- `metrics_comparison.png`
- `confusion_matrix.png`
- `latent_tsne.png`
- `feature_ablation.png`
- `feature_ablation.json`

## How to Reproduce Results (Full Pipeline)

From project root:

```bash
python src/data_pipeline.py
python src/deep_mlp.py
python src/plot_results.py
```

For deterministic behavior, the training script sets a fixed random seed (`42`).

## Demo Notebook

Use [demo_notebook.ipynb](demo_notebook.ipynb) to:

- Run the preprocessing step
- Train the Deep MLP
- Run evaluation and plotting
- Verify key output files
