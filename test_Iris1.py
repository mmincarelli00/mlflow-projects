import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experiment Iris-1")

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Just train the model normally
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# ---- load the logged model and predict ----

# Get the run that autolog just created
run_id = mlflow.last_active_run().info.run_id
model_uri = f"runs:/{run_id}/model"
print("Model URI:", model_uri)

# Load it back from MLflow (generic pyfunc interface)
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Predict on the test set
preds = loaded_model.predict(X_test)

# Create a DataFrame to compare predictions with true labels
names = datasets.load_iris().target_names
results = pd.DataFrame({
    "prediction": preds,
    "prediction_name": names[preds],
    "y_test": y_test,
    "y_test_name": names[y_test],
})

results["correct"] = results["prediction"] == results["y_test"]
print(results)

# 4. Evaluate against the true labels
print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds, average="macro"))
print("Recall   :", recall_score(y_test, preds, average="macro"))
print("F1       :", f1_score(y_test, preds, average="macro"))