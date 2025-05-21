import numpy as np
from sklearn.svm import SVC
from joblib import dump, load
import os
from PIL import Image
from config import Config
from models.mydataset import MyDataset
from services.image_processing import transforming


def start(
    X_train,
    X_val,
    y_train,
    y_val,
    hyperparameters,
    socketio,
):
    socketio.emit("log", {"data": "Starting SVM training..."})
    socketio.emit("log", {"data": "Hyperparameters:"})
    socketio.emit("log", {"data": str(hyperparameters)})
    train_data = MyDataset(X_train, y_train)
    val_data = MyDataset(X_val, y_val)

    kernel = hyperparameters.get("kernel").lower()
    C = hyperparameters.get("c")
    gamma = hyperparameters.get("gamma")
    degree = hyperparameters.get("degree")
    coef0 = hyperparameters.get("coef0")

    X_train_np = []
    y_train_np = []
    for i in range(len(train_data)):
        x, y = train_data[i]
        X_train_np.append(x)
        y_train_np.append(y)

    X_val_np = []
    y_val_np = []
    for i in range(len(val_data)):
        x, y = val_data[i]
        X_val_np.append(x)
        y_val_np.append(y)

    X_train = np.array(X_train_np)
    y_train = np.array(y_train_np)
    X_val = np.array(X_val_np)
    y_val = np.array(y_val_np)

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    if kernel == "poly":
        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            probability=True,
        )
    elif kernel == "sigmoid":
        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            coef0=coef0,
            probability=True,
        )
    elif kernel == "rbf":
        svm_model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
        )
    elif kernel == "linear":
        svm_model = SVC(
            kernel=kernel,
            C=C,
            probability=True,
        )

    socketio.emit(
        "log", {"data": f"Training SVM with {kernel} kernel, C={C}, gamma={gamma}..."}
    )
    svm_model.fit(X_train, y_train)

    train_acc = svm_model.score(X_train, y_train)

    val_acc = svm_model.score(X_val, y_val)

    socketio.emit(
        "log",
        {
            "data": f"Training completed | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        },
    )

    os.makedirs("best-models", exist_ok=True)
    dump(svm_model, "best-models/svm.joblib")
    socketio.emit(
        "log", {"data": f"Saved best model with validation accuracy: {val_acc:.4f}"}
    )
    socketio.emit("log", {"data": "SVM Training Finished!"})


def run_test():
    svm_model = load("best-models/svm.joblib")

    img_path = "static/uploads/test-images/image_name.png"

    img = Image.open(img_path).convert("RGB")
    _, val_transform, _, _ = transforming()
    img_tensor = val_transform(img)
    img_flat = img_tensor.numpy().flatten().reshape(1, -1)
    predicted_class = svm_model.predict(img_flat)[0]
    label = Config.CLASS_MAP[predicted_class]

    return label
