from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Generate 'Borrowers'
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Classifier
clf = LogisticRegression().fit(X_train, y_train)

# 3. Report
print("\n--- Institutional Credit Risk Report ---")
print(classification_report(y_test, clf.predict(X_test)))