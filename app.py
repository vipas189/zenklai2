from flask import Flask, redirect, url_for
from config import Config
from extensions import db, migrate
from flask_socketio import SocketIO
from routes.home_route import register_routes
from models.training_data import TrainingData


app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
db.init_app(app)
migrate.init_app(app, db)


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("home_bp.home"))


register_routes(app, socketio)


if __name__ == "__main__":
    socketio.run(app, debug=True)
