from extensions import db


class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Filename = db.Column(db.String(250), nullable=False)
    Width = db.Column(db.Integer, nullable=False)
    Height = db.Column(db.Integer, nullable=False)
    Roi_X1 = db.Column(db.Integer, nullable=False)
    Roi_Y1 = db.Column(db.Integer, nullable=False)
    Roi_X2 = db.Column(db.Integer, nullable=False)
    Roi_Y2 = db.Column(db.Integer, nullable=False)
    ClassId = db.Column(db.Integer, nullable=False)
