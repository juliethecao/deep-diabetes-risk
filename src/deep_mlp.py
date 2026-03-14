import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json

from plot_utils import plot_models, plot_metrics, plot_confusion, plot_latent_tsne, plot_feature_ablation

FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
BINARY_COLS  = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17]
NUMERIC_COLS = [3, 13, 14, 15, 18, 19, 20]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha = targets * 0.25 + (1 - targets) * 0.75
        gamma = 2.0
        loss = bce * alpha * ((1 - pt) ** gamma)
        return loss.mean()

def choose_loss_func(loss_type, pos_weight):
    if loss_type == "weighted_bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        return FocalLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")

class CategoricalEmbedding(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(1, 4) for _ in range(num_features)
        ])

    def forward(self, x):
        parts = []
        for i, proj in enumerate(self.projections):
            parts.append(proj(x[:, i:i+1]))
        return torch.cat(parts, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        return x + self.block(x)

class DeepMLP(nn.Module):
    def __init__(self, dropout_p=0.3):
        super().__init__()
        self.cat_embed = CategoricalEmbedding(14)
        self.numeric_proj = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
        )
        self.stem = nn.Sequential(
            nn.Linear(120, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(dropout_p),
            ResidualBlock(dropout_p),
            ResidualBlock(dropout_p),
            ResidualBlock(dropout_p),
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_latent=False):
        x_binary  = x[:, BINARY_COLS]
        x_numeric = x[:, NUMERIC_COLS]
        emb  = self.cat_embed(x_binary)
        cont = self.numeric_proj(x_numeric)
        h = torch.cat([emb, cont], dim=1)
        h = self.stem(h)
        h = self.res_blocks(h)
        logit = self.head(h).squeeze(-1)
        if return_latent:
            return logit, h
        return logit

def train_one_epoch(model, loader, optimizer, loss_func, device):
    model.train()
    totalLoss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        outputs = model(x)
        lossOutput = loss_func(outputs, y)
        optimizer.zero_grad()
        lossOutput.backward()
        optimizer.step()
        
        pred = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (pred == y.long()).sum().item()
        bsize = x.size(0)
        totalLoss += lossOutput.item() * bsize
        total += bsize

    loss = totalLoss / total
    acc = correct / total
    return {"loss": loss, "acc": acc}

@torch.no_grad()
def evaluate(model, loader, loss_func, device):
    model.eval()
    totalLoss = 0.0
    total = 0
    all_logits = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        outputs = model(x)
        lossOutput = loss_func(outputs, y)
        bsize = x.size(0)
        totalLoss += lossOutput.item() * bsize
        total += bsize
        all_logits.append(outputs.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)

    loss = totalLoss / total
    acc = accuracy_score(labels, preds)
    metrics = {
        "loss":      loss,
        "acc":       acc,
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc_score(labels, probs),
    }
    return metrics, probs, labels

def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    print(f"Using device: {device}")

    # Load data
    train_data = torch.load("../data_processed/train.pt", weights_only=False)
    val_data   = torch.load("../data_processed/val.pt",   weights_only=False)
    test_data  = torch.load("../data_processed/test.pt",  weights_only=False)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val,   y_val   = val_data["X"],   val_data["y"]
    X_test,  y_test  = test_data["X"],  test_data["y"]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=512)
    test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=512)

    num_healthy  = (y_train == 0).sum().float()
    num_diabetic = (y_train == 1).sum().float()
    pos_weight = (num_healthy / num_diabetic).to(device)
    print(f"Class imbalance weight: {pos_weight.item():.2f}x")
    loss_func = choose_loss_func("weighted_bce", pos_weight)

    model = DeepMLP(dropout_p=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_val_auc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_metrics_history = []

    for epoch in range(1, 30):
        tr = train_one_epoch(model, train_loader, optimizer, loss_func, device)
        val = evaluate(model, val_loader, loss_func, device)[0]
        scheduler.step()

        train_losses.append(tr["loss"])
        val_losses.append(val["loss"])
        val_metrics_history.append(val)

        print(f"Epoch {epoch:3d}/30  |  "
              f"Train Loss: {tr['loss']:.3f}  |  "
              f"Train Acc: {tr['acc']:.3f}  |  "
              f"Val Loss: {val['loss']:.3f}  |  "
              f"Val AUC: {val['roc_auc']:.3f}  |  "
              f"Val F1: {val['f1']:.3f}  |  "
              f"Val Recall: {val['recall']:.3f}")

        if val["roc_auc"] > best_val_auc:
            best_val_auc = val["roc_auc"]
            patience_counter = 0
            torch.save(model.state_dict(), "results/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("results/best_model.pt", weights_only=True))
    te, test_probs, test_labels = evaluate(model, test_loader, loss_func, device)
    test_preds = (test_probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    te["specificity"] = tn / (tn + fp)

    for name, value in te.items():
        print(f"  {name:>15s}: {value:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(test_labels, test_preds)}")
    print(f"\n{classification_report(test_labels, test_preds, target_names=['No Diabetes', 'Diabetes'])}")

    with open("results/test_metrics.json", "w") as f:
        json.dump(te, f, indent=2)

    print("Generating plots...")
    all_probs = {"Deep MLP": test_probs}
    plot_models(test_labels, all_probs, "results", plot_type="roc")
    plot_models(test_labels, all_probs, "results", plot_type="pr")
    plot_models(test_labels, all_probs, "results", plot_type="calibration")
    plot_metrics(test_labels, all_probs, "results")
    plot_confusion(test_labels, test_preds, "results")
    plot_latent_tsne(model, test_loader, device, "results")

    print("Running feature ablation (may take a minute)...")
    drops = plot_feature_ablation(model, X_test, y_test, FEATURE_NAMES, device, "results")
    with open("results/feature_ablation.json", "w") as f:
        json.dump(drops, f, indent=2)

    print("\nAll done! Check the 'results/' folder for all outputs.")

if __name__ == "__main__":
    main()