import os
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from deep_mlp import DeepMLP
from baselines import get_all_baseline_probs
from plot_utils import (
    plot_models, plot_metrics, plot_confusion, plot_latent_tsne,
    plot_feature_ablation, plot_training_curves
)

SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_data():
    train = torch.load("../data_processed/train.pt", weights_only=False)
    test  = torch.load("../data_processed/test.pt",  weights_only=False)
    return train["X"], train["y"], test["X"], test["y"]

def get_deep_mlp_probs(X_test, device):
    model = DeepMLP().to(device)
    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(device))
        return torch.sigmoid(logits).cpu().numpy()

def build_metrics_rows(y_true, all_probs):
    rows = []
    for name, probs in all_probs.items():
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_true, preds),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "Recall": recall_score(y_true, preds, zero_division=0),
            "F1": f1_score(y_true, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, probs),
            "Specificity": specificity,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        })
    return rows

def format_metrics_table(rows):
    headers = [
        "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
        "Specificity", "TP", "TN", "FP", "FN"
    ]
    widths = {
        "Model": max(len("Model"), *(len(r["Model"]) for r in rows)),
        "Accuracy": 8,
        "Precision": 9,
        "Recall": 7,
        "F1": 6,
        "ROC-AUC": 8,
        "Specificity": 11,
        "TP": 6,
        "TN": 6,
        "FP": 6,
        "FN": 6,
    }

    def fmt_float(v, key):
        return f"{v:.4f}".rjust(widths[key])

    def fmt_int(v, key):
        return str(v).rjust(widths[key])

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    lines = [header_line, sep_line]

    for r in rows:
        line = " | ".join([
            r["Model"].ljust(widths["Model"]),
            fmt_float(r["Accuracy"], "Accuracy"),
            fmt_float(r["Precision"], "Precision"),
            fmt_float(r["Recall"], "Recall"),
            fmt_float(r["F1"], "F1"),
            fmt_float(r["ROC-AUC"], "ROC-AUC"),
            fmt_float(r["Specificity"], "Specificity"),
            fmt_int(r["TP"], "TP"),
            fmt_int(r["TN"], "TN"),
            fmt_int(r["FP"], "FP"),
            fmt_int(r["FN"], "FN"),
        ])
        lines.append(line)

    return "\n".join(lines)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_data()
    y_test_np = y_test.numpy()

    print("Running baseline models...")
    all_probs = get_all_baseline_probs(X_train, X_test, y_train)

    print("Getting Deep MLP predictions...")
    all_probs["Deep MLP"] = get_deep_mlp_probs(X_test, device)

    print("Computing evaluation metrics table...")
    metric_rows = build_metrics_rows(y_test_np, all_probs)
    metric_table = format_metrics_table(metric_rows)
    print("\nEvaluation Metrics (Baselines + Deep MLP)")
    print(metric_table)

    metrics_txt_path = os.path.join(SAVE_DIR, "model_metrics_table.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("Evaluation Metrics (Baselines + Deep MLP)\n")
        f.write(metric_table)
        f.write("\n")
    print(f"Saved metrics table to {metrics_txt_path}")

    print("Generating plots...")
    # Calibration, ROC, PR curves
    plot_models(y_test_np, all_probs, SAVE_DIR, plot_type="calibration")
    plot_models(y_test_np, all_probs, SAVE_DIR, plot_type="roc")
    plot_models(y_test_np, all_probs, SAVE_DIR, plot_type="pr")

    # Metrics comparison
    plot_metrics(y_test_np, all_probs, SAVE_DIR)

    # Confusion matrix for Deep MLP only
    mlp_preds = (all_probs["Deep MLP"] >= 0.5).astype(int)
    plot_confusion(y_test_np, mlp_preds, SAVE_DIR)

    # t-SNE and feature ablation for Deep MLP
    # Wrap X_test and y_test in a DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512)
    plot_latent_tsne(DeepMLP().to(device), test_loader, device, SAVE_DIR)
    # Feature names from deep_mlp.py
    from deep_mlp import FEATURE_NAMES
    plot_feature_ablation(DeepMLP().to(device), X_test, y_test, FEATURE_NAMES, device, SAVE_DIR)

    print(f"All plots saved to {SAVE_DIR}/")

if __name__ == "__main__":
    main()