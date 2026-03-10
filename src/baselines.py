from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from data_loader import load_split
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def evaluate_model(model_name, y_true, y_pred, y_probs, plot_calibration=True):
    """
    Print evaluation metrics.
    """
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Specificity:   {specificity:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if plot_calibration:
        prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
        plt.figure(figsize=(6,6))
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        plt.plot([0,1], [0,1], linestyle='--', color='gray')
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve: {model_name}")
        plt.legend()
        plt.grid(True)
        plt.show()


def run_logistic_regression(X_train, X_test, y_train, y_test):

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    evaluate_model("Logistic Regression", y_test, predictions, probabilities)

    return model


def run_knn(X_train, X_test, y_train, y_test, k=5):

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    evaluate_model("KNN", y_test, predictions, probabilities)

    return model


def run_naive_bayes(X_train, X_test, y_train, y_test):

    model = GaussianNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    evaluate_model("Naive Bayes", y_test, predictions, probabilities)

    return model


def run_all_baselines(X_train, X_test, y_train, y_test):
    """
    Run all baseline models.
    """

    models = {}
    models["logistic"] = run_logistic_regression(X_train, X_test, y_train, y_test)
    models["knn"] = run_knn(X_train, X_test, y_train, y_test)
    models["naive_bayes"] = run_naive_bayes(X_train, X_test, y_train, y_test)

    return models

def main():

    # load datasets
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")

    # run baseline models
    run_all_baselines(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()