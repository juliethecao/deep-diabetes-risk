import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from torch.utils.data import DataLoader, TensorDataset

# Import our model
from deep_mlp import DeepMLP

import os
os.makedirs("results_deep_mlp", exist_ok=True)


def load_data():
    """Load data from .pt files."""
    train_data = torch.load("../data_processed/train.pt", weights_only=False)
    test_data  = torch.load("../data_processed/test.pt",  weights_only=False)

    X_train, y_train = train_data["X"], train_data["y"]
    X_test,  y_test  = test_data["X"],  test_data["y"]

    return X_train, y_train, X_test, y_test


def get_baseline_probs(X_train, y_train, X_test):
    """Run all 3 baselines and return their probabilities."""
    # sklearn needs numpy arrays
    X_tr = X_train.numpy()
    y_tr = y_train.numpy()
    X_te = X_test.numpy()

    # Logistic Regression (same settings as Julie's baselines.py)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_tr, y_tr)
    lr_probs = lr.predict_proba(X_te)[:, 1]

    # KNN (same settings as Julie's baselines.py)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    knn_probs = knn.predict_proba(X_te)[:, 1]

    # Naive Bayes (same settings as Julie's baselines.py)
    nb = GaussianNB()
    nb.fit(X_tr, y_tr)
    nb_probs = nb.predict_proba(X_te)[:, 1]

    return lr_probs, knn_probs, nb_probs


def get_deep_mlp_probs(X_test, device):
    """Load the trained Deep MLP and get its probabilities."""
    model = DeepMLP().to(device)
    model.load_state_dict(torch.load("results_deep_mlp/best_model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(X_test.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    return probs


# =====================================================================
# PLOT 1: Individual calibration curve (matches Julie's style exactly)
# =====================================================================

def plot_calibration_single(y_true, probs, model_name, save_dir):
    """Same exact style as Julie's baselines.py calibration plot."""
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve: {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"calibration_{model_name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()


# =====================================================================
# PLOT 2: All models on ONE calibration curve (for easy comparison)
# =====================================================================

def plot_calibration_combined(y_true, all_probs, save_dir):
    """All models on one plot so you can compare directly."""
    plt.figure(figsize=(8, 7))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, probs), color in zip(all_probs.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name, color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve: All Models Compared")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_all_models.png"), dpi=150)
    plt.close()


# =====================================================================
# PLOT 3: All models on ONE ROC curve
# =====================================================================

def plot_roc_combined(y_true, all_probs, save_dir):
    plt.figure(figsize=(8, 7))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, probs), color in zip(all_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})", color=color)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: All Models Compared")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_all_models.png"), dpi=150)
    plt.close()


# =====================================================================
# PLOT 4: All models on ONE Precision-Recall curve
# =====================================================================

def plot_pr_combined(y_true, all_probs, save_dir):
    plt.figure(figsize=(8, 7))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for (name, probs), color in zip(all_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_true, probs)
        plt.plot(rec, prec, label=name, color=color)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: All Models Compared")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_all_models.png"), dpi=150)
    plt.close()


# =====================================================================
# PLOT 5: Bar chart comparing F1, Recall, Precision, AUC across models
# =====================================================================

def plot_metrics_comparison(y_true, all_probs, save_dir):
    """Side-by-side bar chart comparing all models on key metrics."""
    model_names = list(all_probs.keys())
    metrics = {}

    for name, probs in all_probs.items():
        preds = (probs >= 0.5).astype(int)
        metrics[name] = {
            "F1":        f1_score(y_true, preds, zero_division=0),
            "Recall":    recall_score(y_true, preds, zero_division=0),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "ROC-AUC":   roc_auc_score(y_true, probs),
            "Accuracy":  accuracy_score(y_true, preds),
        }

    metric_names = ["F1", "Recall", "Precision", "ROC-AUC", "Accuracy"]
    x = np.arange(len(metric_names))
    width = 0.18  # bar width

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [metrics[name][m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=color)
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: All Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

    # Also print the table
    print("\n" + "=" * 70)
    print("FULL METRICS COMPARISON")
    print("=" * 70)
    header = f"{'Model':>25s}  {'F1':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'AUC':>7s}  {'Acc':>7s}"
    print(header)
    print("-" * 70)
    for name in model_names:
        m = metrics[name]
        print(f"{name:>25s}  {m['F1']:>7.4f}  {m['Recall']:>7.4f}  {m['Precision']:>7.4f}  {m['ROC-AUC']:>7.4f}  {m['Accuracy']:>7.4f}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    X_train, y_train, X_test, y_test = load_data()
    y_test_np = y_test.numpy()

    print("Running baselines...")
    lr_probs, knn_probs, nb_probs = get_baseline_probs(X_train, y_train, X_test)

    print("Loading trained Deep MLP...")
    mlp_probs = get_deep_mlp_probs(X_test, device)

    # All models in one dict (keeps plotting code clean)
    all_probs = {
        "Logistic Regression": lr_probs,
        "KNN": knn_probs,
        "Naive Bayes": nb_probs,
        "Deep MLP": mlp_probs,
    }

    # Individual calibration curves (same style as Julie's)
    print("Generating individual calibration curves...")
    for name, probs in all_probs.items():
        plot_calibration_single(y_test_np, probs, name, "results_deep_mlp")

    # Combined comparison plots
    print("Generating combined comparison plots...")
    plot_calibration_combined(y_test_np, all_probs, "results_deep_mlp")
    plot_roc_combined(y_test_np, all_probs, "results_deep_mlp")
    plot_pr_combined(y_test_np, all_probs, "results_deep_mlp")
    plot_metrics_comparison(y_test_np, all_probs, "results_deep_mlp")

    print(f"\nAll plots saved to results_deep_mlp/")
    print("Individual:  calibration_logistic_regression.png")
    print("             calibration_knn.png")
    print("             calibration_naive_bayes.png")
    print("             calibration_deep_mlp.png")
    print("Combined:    calibration_all_models.png")
    print("             roc_all_models.png")
    print("             pr_all_models.png")
    print("             metrics_comparison.png")


if __name__ == "__main__":
    main()