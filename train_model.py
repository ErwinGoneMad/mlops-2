import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    mlflow.set_experiment("iris_classification")

    with mlflow.start_run():
        X, y = datasets.load_iris(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        params = {
            "solver": "liblinear",
            "max_iter": 2000,
            "multi_class": "ovr",
            "random_state": 8888,
        }

        mlflow.log_params(params)

        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        mlflow.sklearn.log_model(
            lr, "model", registered_model_name="iris_logistic_regression"
        )


if __name__ == "__main__":
    train_model()
