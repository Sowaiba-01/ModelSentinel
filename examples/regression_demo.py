"""Regression flavour of the ModelSentinel workflow."""
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import modelsentinel as ms


def main() -> None:
    X, y = make_regression(n_samples=1500, n_features=6, noise=12.0, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = Ridge().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    monitor = ms.Monitor(task="regression", name="Ridge-demo")
    print("Evaluation:", monitor.evaluate(y_test, y_pred))
    print("Health:", monitor.health_score())


if __name__ == "__main__":
    main()
