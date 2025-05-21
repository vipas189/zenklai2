def early_stop(model, history, early_stop_num, socketio):
    if history.get("epoch_val_acc") > history.get("best_val_accuracy"):
        history["best_val_accuracy"] = history.get("epoch_val_acc")
        history["best_model_wts"] = model.state_dict()
        history["not_improved_val_acc"] = 0

    history["not_improved_val_acc"] += 1

    if history["not_improved_val_acc"] == early_stop_num:
        history["not_improved_val_acc"] = 0
        socketio.emit(
            "log",
            {"data": "Model stopped, no more improvement in validation accuracy"},
        )
        return True
