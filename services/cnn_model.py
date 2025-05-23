import torch
from torch import nn
from torch.utils.data import DataLoader
from services.early_stop import early_stop
from models.simplecnn import SimpleCNN
from models.mydataset import MyDataset
from PIL import Image
from config import Config
from services.image_processing import transforming
from services.diagram import plot_training
import tensorflow as tf
import numpy as np


def start(
    X_train,
    X_val,
    y_train,
    y_val,
    hyperparameters,
    socketio,
):
    socketio.emit(
        "log",
        {"data": "Starting CNN..."},
    )
    socketio.emit(
        "log",
        {"data": "Hyperparameters:"},
    )
    socketio.emit(
        "log",
        {"data": str(hyperparameters)},
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_scores": [],
        "epoch_train_loss": 0,
        "epoch_train_acc": 0,
        "running_train_loss": 0,
        "correct_train": 0,
        "total_train": 0,
        "epoch_val_loss": 0,
        "epoch_val_acc": 0,
        "running_val_loss": 0,
        "correct_val": 0,
        "total_val": 0,
        "best_val_accuracy": 0,
        "best_val_loss": float("inf"),
        "not_improved_val_acc": 0,
        "best_model_wts": None,
    }
    epochs = hyperparameters.get("epochs")
    batch_size = hyperparameters.get("batch_size")
    learning_rate = hyperparameters.get("learning_rate")
    opt_name = hyperparameters.get("optimizer").lower()
    weight_decay = hyperparameters.get("weight_decay")
    dropout_rate = hyperparameters.get("dropout_rate")
    early_stop_num = hyperparameters.get("early_stop")

    train_data = MyDataset(X_train, y_train, augment=True)
    val_data = MyDataset(X_val, y_val, validation=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(set(y_train)), dropout=dropout_rate).to(device)

    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, loss_fn, optimizer, train_loader, history, device, hyperparameters)
        val(model, loss_fn, val_loader, history, device, hyperparameters)
        socketio.emit(
            "log",
            {
                "data": f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {history.get('epoch_train_loss'):.4f} | Train Acc: {history.get('epoch_train_acc'):.4f} | "
                f"Val Loss: {history.get('epoch_val_loss'):.4f} | Val Acc: {history.get('epoch_val_acc'):.4f} - Best: {history.get('best_val_accuracy'):.4f}"
            },
        )
        if history.get("epoch_val_acc") >= history.get("best_val_accuracy"):
            torch.save(model, f"best-models/cnn.pth")
        if early_stop(model, history, early_stop_num, socketio):
            print(f"Early stopping at epoch {epoch+1}")
            break
    plot_training(history)
    socketio.emit(
        "log",
        {"data": "Finished Training!"},
    )


def train(model, loss_fn, optimizer, train_loader, history, device, hyperparameters):
    history["running_train_loss"] = 0.0
    history["correct_train"] = 0
    history["total_train"] = 0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        history["running_train_loss"] += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        history["total_train"] += labels.size(0)
        history["correct_train"] += (predicted == labels).sum().item()

    if history["total_train"] > 0:
        history["epoch_train_loss"] = history.get("running_train_loss") / history.get(
            "total_train"
        )
        history["epoch_train_acc"] = history.get("correct_train") / history.get(
            "total_train"
        )
    else:
        history["epoch_train_loss"] = 0.0
        history["epoch_train_acc"] = 0.0

    history["train_loss"].append(history.get("epoch_train_loss"))
    history["train_acc"].append(history.get("epoch_train_acc"))


def val(model, loss_fn, val_loader, history, device, hyperparameters):
    history["running_val_loss"] = 0.0
    history["correct_val"] = 0
    history["total_val"] = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            history["running_val_loss"] += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            history["total_val"] += labels.size(0)
            history["correct_val"] += (predicted == labels).sum().item()

    if history["total_val"] > 0:
        history["epoch_val_loss"] = history.get("running_val_loss") / history.get(
            "total_val"
        )
        history["epoch_val_acc"] = history.get("correct_val") / history.get("total_val")
    else:
        history["epoch_val_loss"] = 0.0
        history["epoch_val_acc"] = 0.0

    history["val_loss"].append(history.get("epoch_val_loss"))
    history["val_acc"].append(history.get("epoch_val_acc"))


def run_test():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load("best-models/cnn.pth", weights_only=False).to(device)
    # model.eval()

    # img_path = "static/uploads/test-images/image_name.png"
    # img = Image.open(img_path).convert("RGB")
    # _, val_transform, _, _ = transforming()
    # input_tensor = val_transform(img).unsqueeze(0).to(device)

    # with torch.no_grad():
    #     output = model(input_tensor)
    #     _, predicted = torch.max(output, 1)
    #     label = Config.CLASS_MAP[predicted.item()]

    # return label

    model = tf.keras.models.load_model("best-models/cnn.keras")
    img = Image.open("static/uploads/test-images/image_name.png").convert("RGB")
    img = img.resize((64, 64))
    img = np.expand_dims(img, axis=0)
    img = np.array(img, dtype=np.float32) / 255.0

    preds = model.predict(img)
    predicted_class = np.argmax(preds, axis=1)[0]
    label = Config.CLASS_MAP[predicted_class]

    return label
