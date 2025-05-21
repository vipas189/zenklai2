from config import Config


def allowed_file(filename, is_image=False):
    ext = filename.strip().rsplit(".", 1)[-1].lower() if "." in filename else ""
    if is_image:
        return ext in Config.IMAGE_ALLOWED_EXTENSIONS
    return ext in Config.DATABASE_ALLOWED_EXTENSIONS
