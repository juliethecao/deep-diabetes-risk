from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

BASELINE_MODELS = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("KNN", KNeighborsClassifier(n_neighbors=5)),
    ("Naive Bayes", GaussianNB()),
]

def get_all_baseline_probs(X_train, X_test, y_train):
    probs_dict = {}
    for name, model in BASELINE_MODELS:
        model.fit(X_train, y_train)
        probs_dict[name] = model.predict_proba(X_test)[:, 1]
    return probs_dict