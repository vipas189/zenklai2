from sklearn.model_selection import train_test_split
from extensions import db
from models.training_data import TrainingData
import numpy as np


def data_split():
    data = db.session.execute(db.select(TrainingData)).scalars().all()
    X = [(e.Filename, e.Roi_X1, e.Roi_Y1, e.Roi_X2, e.Roi_Y2) for e in data]
    y = [e.ClassId for e in data]

    X_sampled, _, y_sampled, _ = train_test_split(
        X, y, train_size=0.1, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_sampled, y_sampled, test_size=0.15, stratify=y_sampled, random_state=42
    )
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X, y, test_size=0.15, stratify=y, random_state=42
    # )
    return X_train, X_val, y_train, y_val
