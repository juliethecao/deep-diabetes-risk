import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

# =================== Helpers ===================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# =================== Training curves ===================

def plot_training_curves(train_losses, val_losses, val_metrics_history, save_dir):
    save_dir = ensure_dir(save_dir)
    epochs = range(1, len(train_losses)+1)
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss"); axes[0].legend()

    for key in ["f1", "recall", "precision"]:
        axes[1].plot(epochs, [m[key] for m in val_metrics_history], label=key.capitalize())
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics"); axes[1].legend()

    axes[2].plot(epochs, [m["roc_auc"] for m in val_metrics_history], color="purple")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("ROC-AUC")
    axes[2].set_title("Validation ROC-AUC")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()

# =================== Calibration / ROC / PR ===================

def plot_models(y_true, all_probs, save_dir, plot_type="calibration"):
    save_dir = ensure_dir(save_dir)

    if plot_type in ["calibration", "roc", "pr"]:
        plt.figure(figsize=(8,7))
        for name, probs in all_probs.items():
            if plot_type == "calibration":
                prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
                plt.plot(prob_pred, prob_true, marker="o", label=name)
                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title("Calibration Curve" if len(all_probs)==1 else "Calibration Curve: All Models")
                plt.plot([0,1],[0,1], linestyle="--", color="gray")
            elif plot_type == "roc":
                fpr, tpr, _ = roc_curve(y_true, probs)
                auc = roc_auc_score(y_true, probs)
                plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title("ROC Curve" if len(all_probs)==1 else "ROC Curve: All Models")
            elif plot_type == "pr":
                prec, rec, _ = precision_recall_curve(y_true, probs)
                plt.plot(rec, prec, label=name)
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title("Precision-Recall Curve" if len(all_probs)==1 else "PR Curve: All Models")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{plot_type}_all_models.png"), dpi=150)
        plt.close()

# =================== Metrics comparison ===================

def plot_metrics(y_true, all_probs, save_dir):
    save_dir = ensure_dir(save_dir)
    metric_names = ["F1","Recall","Precision","ROC-AUC","Accuracy"]
    x = np.arange(len(metric_names))
    width = 0.18
    metrics = {}
    model_names = list(all_probs.keys())

    for name, probs in all_probs.items():
        preds = (probs >= 0.5).astype(int)
        metrics[name] = {
            "F1": f1_score(y_true, preds, zero_division=0),
            "Recall": recall_score(y_true, preds, zero_division=0),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, probs),
            "Accuracy": accuracy_score(y_true, preds),
        }

    fig, ax = plt.subplots(figsize=(12,6))
    for i, name in enumerate(model_names):
        values = [metrics[name][m] for m in metric_names]
        bars = ax.bar(x + i*width, values, width, label=name)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Score"); ax.set_title("Model Comparison: All Metrics")
    ax.set_xticks(x + width*1.5); ax.set_xticklabels(metric_names)
    ax.set_ylim(0,1.15); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150)
    plt.close()

# =================== Confusion matrix ===================

def plot_confusion(labels, preds, save_dir):
    save_dir = ensure_dir(save_dir)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i,j] > cm.max()/2 else "black"
            ax.text(j,i,f"{cm[i,j]}", ha="center", va="center", fontsize=14, color=color)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Diabetes","Diabetes"])
    ax.set_yticklabels(["No Diabetes","Diabetes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

# =================== t-SNE ===================

def plot_latent_tsne(model, loader, device, save_dir):
    save_dir = ensure_dir(save_dir)
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            _, hidden = model(X_batch.to(device), return_latent=True)
            latents.append(hidden.cpu()); labels.append(y_batch)
    latents = torch.cat(latents).numpy(); labels = torch.cat(labels).numpy()
    if len(latents) > 5000:
        idx = np.random.choice(len(latents), 5000, replace=False)
        latents = latents[idx]; labels = labels[idx]
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(latents)
    fig, ax = plt.subplots(figsize=(8,6))
    for cls, color, name in [(0,"tab:blue","No Diabetes"),(1,"tab:red","Diabetes")]:
        mask = labels==cls
        ax.scatter(coords[mask,0], coords[mask,1], c=color, label=name, alpha=0.4, s=8)
    ax.set_title("t-SNE of Latent Representations"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latent_tsne.png"), dpi=150)
    plt.close()

# =================== Feature ablation ===================

def plot_feature_ablation(model, X_test, y_test, feature_names, device, save_dir):
    save_dir = ensure_dir(save_dir)
    model.eval()
    with torch.no_grad():
        base_probs = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
    base_auc = roc_auc_score(y_test.numpy(), base_probs)
    drops = {}
    for i, name in enumerate(feature_names):
        X_mod = X_test.clone(); X_mod[:, i] = 0.0
        with torch.no_grad():
            probs = torch.sigmoid(model(X_mod.to(device))).cpu().numpy()
        drops[name] = base_auc - roc_auc_score(y_test.numpy(), probs)
    # plot
    sorted_feats = sorted(drops.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feats]; vals = [f[1] for f in sorted_feats]
    fig, ax = plt.subplots(figsize=(10,6))
    colors = ["tab:red" if v>0 else "tab:blue" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("ROC-AUC Drop"); ax.set_title("Feature Ablation — Importance")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_ablation.png"), dpi=150)
    plt.close()
    return drops