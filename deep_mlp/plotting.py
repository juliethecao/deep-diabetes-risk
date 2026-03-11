import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE


def plot_training_curves(train_losses, val_losses, val_metrics_history, save_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    for key in ["f1", "recall", "precision"]:
        axes[1].plot(epochs, [m[key] for m in val_metrics_history], label=key.capitalize())
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    axes[2].plot(epochs, [m["roc_auc"] for m in val_metrics_history], color="purple")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].set_title("Validation ROC-AUC")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def plot_roc_and_pr(labels, probs, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve (Test Set)")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(labels, probs)
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve (Test Set)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_pr_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14, color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_yticklabels(["No Diabetes", "Diabetes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_calibration(labels, probs, save_dir):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_curve.png"), dpi=150)
    plt.close()


def plot_latent_tsne(model, loader, device, save_dir):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            _, hidden = model(X_batch.to(device), return_latent=True)
            latents.append(hidden.cpu())
            labels.append(y_batch)

    latents = torch.cat(latents).numpy()
    labels = torch.cat(labels).numpy()

    if len(latents) > 5000:
        idx = np.random.choice(len(latents), 5000, replace=False)
        latents = latents[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color, name in [(0, "tab:blue", "No Diabetes"), (1, "tab:red", "Diabetes")]:
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1], c=color, label=name, alpha=0.4, s=8)
    ax.set_title("t-SNE of Latent Representations (Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latent_tsne.png"), dpi=150)
    plt.close()


def plot_feature_ablation(model, X_test, y_test, feature_names, device, save_dir):
    model.eval()

    with torch.no_grad():
        base_probs = torch.sigmoid(model(X_test.to(device))).cpu().numpy()
    base_auc = roc_auc_score(y_test.numpy(), base_probs)

    drops = {}
    for i, name in enumerate(feature_names):
        X_modified = X_test.clone()
        X_modified[:, i] = 0.0

        with torch.no_grad():
            probs = torch.sigmoid(model(X_modified.to(device))).cpu().numpy()
        auc = roc_auc_score(y_test.numpy(), probs)
        drops[name] = base_auc - auc

    sorted_feats = sorted(drops.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_feats]
    vals  = [f[1] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("ROC-AUC Drop (higher = more important)")
    ax.set_title("Feature Ablation — Importance by AUC Drop (Test Set)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_ablation.png"), dpi=150)
    plt.close()

    return drops