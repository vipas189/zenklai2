from flask import (
    request,
    render_template,
    redirect,
    url_for,
    flash,
    current_app,
    Blueprint,
    Response,
)
import pandas as pd
from models.training_data import TrainingData
import os
from werkzeug.utils import secure_filename
from extensions import db
import pandas as pd
from services.allowed_file import allowed_file
import shutil
from werkzeug.exceptions import RequestEntityTooLarge
import zipfile
import io
import services.cnn_model as cnn_model
import services.svm_model as svm_model
import services.vit_model as vit_model
from services.data_split import data_split
from config import Config
import glob
from PIL import Image
from threading import Thread


home_bp = Blueprint("home_bp", __name__)

# Store the socketio instance (or pass it directly to the route functions)
# This is a simple way, but be mindful of global state in a complex app
_socketio = None
_app_instance = None


def register_routes(app, socketio_instance):
    """Registers the home routes and stores the socketio instance."""
    global _socketio, _app_instance
    _socketio = socketio_instance
    _app_instance = app
    app.register_blueprint(home_bp)


@home_bp.route("/")
def home():
    db_rows = db.session.execute(db.select(TrainingData)).scalars().all()
    for index, db_row in enumerate(db_rows):
        if index > 2:
            flash(f"+{len(db_rows) - 3} images  ", category="image_count")
            break
        flash(f"{db_row.Filename}, ", category="image_count")

    return render_template("home.html")


@home_bp.route("/upload/folder", methods=["POST"])
def upload_folder():
    try:
        upload_dir = os.path.join(current_app.root_path, "static/uploads")
        os.makedirs(upload_dir, exist_ok=True)
        zip_file = request.files.get("train_zip")
        if not zip_file or zip_file.filename == "":
            flash("No zip file selected", "error")
            return redirect(url_for("home_bp.home"))

        if not zip_file.filename.lower().endswith(".zip"):
            flash("Invalid file type. Please upload a .zip file.", "error")
            return redirect(url_for("home_bp.home"))

        img_folder = os.path.join(upload_dir, "images")
        os.makedirs(img_folder, exist_ok=True)
        csv_processed = 0
        images_processed = 0
        zip_content = io.BytesIO(zip_file.read())

        try:
            with zipfile.ZipFile(zip_content, "r") as zip_ref:
                zip_filenames = zip_ref.namelist()

                for filename_in_zip in zip_filenames:
                    sanitized_filename_in_zip = secure_filename(filename_in_zip)

                    if sanitized_filename_in_zip.endswith("/"):
                        continue

                    subfolder_path_in_zip = os.path.dirname(filename_in_zip)
                    base_filename_in_zip = os.path.basename(filename_in_zip)
                    last_subfolder_name = os.path.basename(subfolder_path_in_zip)

                    new_name = (
                        f"{last_subfolder_name}_{base_filename_in_zip}"
                        if last_subfolder_name
                        else base_filename_in_zip
                    )
                    if allowed_file(base_filename_in_zip):
                        target_file_path_base = upload_dir
                    elif allowed_file(base_filename_in_zip, is_image=True):
                        target_file_path_base = img_folder
                    else:
                        print(
                            f"Skipping unsupported file type in zip: {filename_in_zip}"
                        )
                        continue

                    target_file_path = os.path.join(target_file_path_base, new_name)
                    intended_base = os.path.realpath(target_file_path_base)
                    resolved_target = os.path.realpath(target_file_path)

                    if not resolved_target.startswith(intended_base):
                        print(f"Path traversal attempt blocked for: {filename_in_zip}")
                        flash(
                            f"Blocked potentially unsafe path in zip: {filename_in_zip}",
                            "error",
                        )
                        continue

                    try:
                        with zip_ref.open(filename_in_zip, "r") as internal_file:
                            file_content = internal_file.read()

                        if allowed_file(base_filename_in_zip):
                            temp_csv_path = os.path.join(
                                upload_dir, f"temp_zip_{new_name}"
                            )
                            with open(temp_csv_path, "wb") as temp_csv_file:
                                temp_csv_file.write(file_content)

                            try:
                                for chunk in pd.read_csv(
                                    temp_csv_path, delimiter=";", chunksize=5000
                                ):
                                    chunk["Filename"] = (
                                        f"{last_subfolder_name}_"
                                        if last_subfolder_name
                                        else ""
                                    ) + chunk["Filename"]

                                    chunk.columns = chunk.columns.str.replace(".", "_")

                                    existing_filenames = {
                                        row[0]
                                        for row in db.session.execute(
                                            db.select(TrainingData.Filename).where(
                                                TrainingData.Filename.in_(
                                                    chunk["Filename"].tolist()
                                                )
                                            )
                                        )
                                    }

                                    chunk = chunk[
                                        ~chunk["Filename"].isin(existing_filenames)
                                    ]

                                    if not chunk.empty:
                                        chunk.to_sql(
                                            "training_data",
                                            con=db.engine,
                                            if_exists="append",
                                            index=False,
                                        )
                                        csv_processed += len(chunk)

                            finally:
                                if os.path.exists(temp_csv_path):
                                    os.remove(temp_csv_path)

                        elif allowed_file(base_filename_in_zip, is_image=True):
                            with open(target_file_path, "wb") as img_file:
                                img_file.write(file_content)
                            images_processed += 1

                    except zipfile.BadZipFile:
                        flash("Invalid zip file format.", "error")
                        return redirect(url_for("home_bp.home"))
                    except Exception as e:
                        flash(
                            f"Error processing file from zip ({filename_in_zip}): {str(e)}",
                            "error",
                        )
                        import traceback

                        print(f"Error processing {filename_in_zip}:")
                        traceback.print_exc()
                        pass

        except zipfile.BadZipFile:
            flash("The uploaded file is not a valid zip file.", "error")
            return redirect(url_for("home_bp.home"))
        except Exception as e:
            flash(
                f"An error occurred while processing the zip file: {str(e)}",
                "error",
            )
            import traceback

            print("Unhandled exception during zip processing:")
            traceback.print_exc()
            return redirect(url_for("home_bp.home"))

        db.session.commit()

        db_rows = db.session.execute(db.select(TrainingData)).scalars().all()
        expected_image_filenames_in_db = {row.Filename for row in db_rows}
        actual_images_on_disk = set(os.listdir(img_folder))

        removed_db_entries = 0
        for db_row in db_rows:
            if db_row.Filename not in actual_images_on_disk:
                print(f"Removing DB entry for missing image: {db_row.Filename}")
                db.session.delete(db_row)
                removed_db_entries += 1

        removed_images = 0
        for filename_on_disk in actual_images_on_disk:
            if filename_on_disk not in expected_image_filenames_in_db:
                file_path = os.path.join(img_folder, filename_on_disk)
                print(f"Removing orphaned image: {file_path}")
                try:
                    os.remove(file_path)
                    removed_images += 1
                except PermissionError:
                    print(
                        f"Could not delete orphaned image (permission error): {file_path}"
                    )

        db.session.commit()

        # flash(
        #     f"Processed {csv_processed} CSV records and {images_processed} images from zip. Removed {removed_db_entries} database entries and {removed_images} orphaned images.",
        #     "image_count",
        # )
        return redirect(url_for("home_bp.home"))

    except RequestEntityTooLarge:
        flash(
            "The uploaded zip file is too large. Please try uploading a smaller zip file or increasing the MAX_CONTENT_LENGTH in your Flask configuration.",
            "error",
        )
        return redirect(url_for("home_bp.home"))
    except Exception as e:
        flash(f"An unhandled error occurred during zip processing: {str(e)}", "error")
        import traceback

        print("Unhandled exception during zip upload:")
        traceback.print_exc()
        return redirect(url_for("home_bp.home"))


@home_bp.route("/delete/upload", methods=["POST"])
def delete_upload():
    db.session.execute(db.delete(TrainingData))
    db.session.commit()
    img_folder = os.path.join(current_app.root_path, "static/uploads/images")
    shutil.rmtree(img_folder)

    return redirect(url_for("home_bp.home"))


@home_bp.route("/start/training", methods=["POST"])
def start_training():
    model_type = request.form.get("model")
    hyperparameters = {"model": model_type}

    if model_type == "cnn-hyperparams":
        hyperparameters.update(
            {
                "epochs": request.form.get("cnn-epochs", type=int),
                "batch_size": request.form.get("cnn-batch-size", type=int),
                "learning_rate": request.form.get("cnn-learning-rate", type=float),
                "optimizer": request.form.get("cnn-optimizer"),
                "weight_decay": request.form.get("cnn-weight-decay", type=float),
                "dropout_rate": request.form.get("cnn-dropout-rate", type=float),
                "early_stop": request.form.get("cnn-early-stop", type=int),
            }
        )
    elif model_type == "svm-hyperparams":
        hyperparameters.update(
            {
                "c": request.form.get("svm-c", type=float),
                "kernel": request.form.get("svm-kernel"),
                "gamma": request.form.get("svm-gamma"),
                "degree": request.form.get("svm-degree", type=int),
                "coef0": request.form.get("svm-coef0", type=float),
            }
        )
    elif model_type == "vit-hyperparams":
        hyperparameters.update(
            {
                "epochs": request.form.get("vit-epochs", type=int),
                "batch_size": request.form.get("vit-batch-size", type=int),
                "early_stop": request.form.get("vit-early-stop", type=int),
                "learning_rate": request.form.get("vit-learning-rate", type=float),
                "weight_decay": request.form.get("vit-weight-decay", type=float),
                "dropout_rate": request.form.get("vit-dropout-rate", type=float),
            }
        )

    def background_train_task(hp_dict):
        with _app_instance.app_context():
            current_model_type = hp_dict.get("model")
            X_train, X_val, y_train, y_val = data_split()
            if current_model_type == "cnn-hyperparams":

                cnn_model.start(
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    hp_dict,
                    _socketio,
                )
            elif current_model_type == "svm-hyperparams":
                svm_model.start(
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    hp_dict,
                    _socketio,
                )
            elif current_model_type == "vit-hyperparams":
                vit_model.start(
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    hp_dict,
                    _socketio,
                )

    thread = Thread(target=background_train_task, args=(hyperparameters,))
    thread.start()
    return Response(status=204)


@home_bp.route("/test/upload/image", methods=["POST"])
def test_upload_image():
    file = request.files["test-file-input"]

    if file.filename == "":
        flash("No selected file", "error")
        return redirect(url_for("home_bp.home"))

    if not allowed_file(file.filename, is_image=True):
        allowed_types = ", ".join(Config.IMAGE_ALLOWED_EXTENSIONS)
        flash(f"File type not allowed. Allowed types are: {allowed_types}", "error")
        return redirect(url_for("home_bp.home"))

    if file:
        try:
            folder = os.path.join("static", "uploads", "test-images")
            for f in glob.glob(os.path.join(folder, "image_name.*")):
                os.remove(f)

            # Always save as PNG
            new_name = "image_name.png"
            filepath = os.path.join(folder, new_name)

            img = Image.open(file.stream)
            img.save(filepath, format="PNG")

            flash(f"File uploaded as '{new_name}'.", "success")
            return redirect(url_for("home_bp.home"))
        except Exception as e:
            print(f"Error saving file: {e}")
            flash("An error occurred while uploading the image.", "error")
            return redirect(url_for("home_bp.home"))

    flash("Unexpected error during file upload.", "error")
    return redirect(url_for("home_bp.home"))


@home_bp.route("/run/test", methods=["POST"])
def run_test():
    model = request.form.get("model")
    if model == "cnn":
        flash(cnn_model.run_test(), category="traffic-sign")
    elif model == "svm":
        flash(svm_model.run_test(), category="traffic-sign")
    elif model == "vit":
        flash(vit_model.run_test(), category="traffic-sign")
    return redirect(url_for("home_bp.home"))
