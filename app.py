import io
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.routing import BuildError

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - handled by API errors.
    tf = None

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# Variables
NAV_ITEMS = [
    {"label": "Home", "endpoint": "index"},
    {"label": "About", "endpoint": "about"},
]

MODELS_DIR = Path(app.root_path) / "models"
HISTORIES_DIR = Path(app.root_path) / "histories"
STATIC_DIR = Path(app.root_path) / "static"
MODEL_PATH = MODELS_DIR / "model.keras"
HISTORY_FILE_CANDIDATES = [
    "model-history.np",
    "model-history.npy",
    "model-history.npz",
]
SAMPLE_IMAGE_DIR_CANDIDATES = [
    STATIC_DIR / "imgs",
    STATIC_DIR / "images" / "samples",
]
SAMPLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPLOAD_IMAGE_SUFFIXES = {".jpg", ".jpeg"}

MODEL_CACHE = {
    "path": None,
    "mtime": None,
    "model": None,
}


#-----------------
def _build_nav_items():
    """Validate endpoints so broken nav entries do not render."""
    safe_items = []
    for item in NAV_ITEMS:
        endpoint = item.get("endpoint")
        try:
            url_for(endpoint)
        except BuildError:
            app.logger.warning("Skipping nav item with unknown endpoint: %s", endpoint)
            continue

        safe_items.append(
            {
                "label": item.get("label", endpoint.title()),
                "endpoint": endpoint,
                "href": url_for(endpoint),
            }
        )
    return safe_items


def _short_exception(error: Exception, limit: int = 220):
    """Keep API/debug messages readable when deep stack traces are thrown."""
    message = f"{error.__class__.__name__}: {str(error)}"
    if len(message) <= limit:
        return message
    return f"{message[:limit - 3]}..."


def _build_api_error(error_type, message, status_code, *, path=None, details=None):
    error_payload = {
        "type": error_type,
        "message": message,
    }
    if path:
        error_payload["path"] = path
    if details:
        error_payload["details"] = details
    return {"error": error_payload}, status_code


def _json_error_response(error, data):
    payload, status_code = _build_api_error(
        error["type"],
        error["message"],
        error["status_code"],
        path=error.get("path"),
        details=error.get("details"),
    )
    payload["data"] = data
    return jsonify(payload), status_code


def _resolve_history_file():
    for file_name in HISTORY_FILE_CANDIDATES:
        path = HISTORIES_DIR / file_name
        if path.exists():
            return path

    wildcard_matches = sorted(HISTORIES_DIR.glob("model-history*"))
    return wildcard_matches[0] if wildcard_matches else None


def _coerce_series_to_float_list(value):
    if isinstance(value, np.ndarray):
        flat_values = value.flatten().tolist()
    elif isinstance(value, (list, tuple)):
        flat_values = list(value)
    elif isinstance(value, (int, float, np.number)):
        flat_values = [value]
    else:
        return []

    cleaned = []
    for item in flat_values:
        try:
            cleaned.append(float(item))
        except (TypeError, ValueError):
            continue
    return cleaned


def _normalize_history_payload(payload):
    if isinstance(payload, dict):
        return payload

    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            return {key: payload[key] for key in payload.files}
        finally:
            payload.close()

    if isinstance(payload, np.ndarray):
        if payload.dtype == object:
            if payload.shape == ():
                maybe_dict = payload.item()
            elif payload.size == 1:
                maybe_dict = payload.reshape(()).item()
            else:
                maybe_dict = payload.tolist()
            if isinstance(maybe_dict, dict):
                return maybe_dict
            return {}
        return {"metric": payload.tolist()}

    return {}


def _build_model_file_stats():
    if not MODEL_PATH.exists():
        return {"model_file_status": "missing"}

    stat = MODEL_PATH.stat()
    return {
        "model_file_status": "available",
        "model_file_name": MODEL_PATH.name,
        "model_file_size_mb": round(stat.st_size / (1024 * 1024), 3),
        "model_file_updated": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _load_history_dict():
    history_path = _resolve_history_file()
    if history_path is None:
        return None, None, {
            "type": "missing_history_file",
            "message": "No model history file found in histories/.",
            "status_code": 404,
            "path": str(HISTORIES_DIR),
        }

    try:
        raw_payload = np.load(history_path, allow_pickle=True)
    except PermissionError as error:
        return None, None, {
            "type": "history_file_permission_error",
            "message": "History file exists but cannot be read due to file permissions.",
            "status_code": 500,
            "path": str(history_path),
            "details": _short_exception(error),
        }
    except OSError as error:
        return None, None, {
            "type": "history_file_read_error",
            "message": "History file exists but could not be read.",
            "status_code": 500,
            "path": str(history_path),
            "details": _short_exception(error),
        }
    except Exception as error:
        return None, None, {
            "type": "history_file_parse_error",
            "message": "History file was read but could not be parsed.",
            "status_code": 500,
            "path": str(history_path),
            "details": _short_exception(error),
        }

    history = _normalize_history_payload(raw_payload)
    if not isinstance(history, dict) or not history:
        return None, None, {
            "type": "invalid_history_format",
            "message": f"History payload in {history_path.name} is not a usable metrics dictionary.",
            "status_code": 500,
            "path": str(history_path),
        }

    numeric_history = {}
    for metric_name, values in history.items():
        series = _coerce_series_to_float_list(values)
        if series:
            numeric_history[metric_name] = series

    if not numeric_history:
        return None, None, {
            "type": "invalid_history_content",
            "message": f"No numeric metrics found in {history_path.name}.",
            "status_code": 500,
            "path": str(history_path),
        }

    return numeric_history, history_path, None


def _load_stats_from_history():
    history, history_path, error = _load_history_dict()
    if error:
        return None, error

    stats = {}
    epochs = 0

    for metric_name, series in history.items():
        stats[metric_name] = round(series[-1], 4)
        epochs = max(epochs, len(series))

    if "val_accuracy" in history:
        stats["best_val_accuracy"] = round(max(history["val_accuracy"]), 4)
    if "val_loss" in history:
        stats["best_val_loss"] = round(min(history["val_loss"]), 4)

    stats["epochs"] = epochs
    stats["stats_source"] = f"histories/{history_path.name}"
    stats.update(_build_model_file_stats())
    return stats, None


def _list_sample_images():
    images = []
    for directory in SAMPLE_IMAGE_DIR_CANDIDATES:
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in SAMPLE_IMAGE_SUFFIXES:
                continue
            images.append(path)
    return sorted(images)


def _get_cached_model():
    if tf is None:
        return None, {
            "type": "dependency_missing",
            "message": "TensorFlow is unavailable in this environment.",
            "status_code": 500,
        }

    if not MODEL_PATH.exists():
        return None, {
            "type": "missing_model_file",
            "message": "Model file was not found.",
            "status_code": 404,
            "path": str(MODEL_PATH),
        }

    current_mtime = MODEL_PATH.stat().st_mtime
    cached = MODEL_CACHE
    if (
        cached["model"] is not None
        and cached["path"] == str(MODEL_PATH)
        and cached["mtime"] == current_mtime
    ):
        return cached["model"], None

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except PermissionError as error:
        return None, {
            "type": "model_file_permission_error",
            "message": "Model file exists but cannot be read due to file permissions.",
            "status_code": 500,
            "path": str(MODEL_PATH),
            "details": _short_exception(error),
        }
    except OSError as error:
        return None, {
            "type": "model_file_read_error",
            "message": "Model file exists but could not be read.",
            "status_code": 500,
            "path": str(MODEL_PATH),
            "details": _short_exception(error),
        }
    except Exception as error:
        return None, {
            "type": "model_deserialization_error",
            "message": "Model file was read but could not be deserialized by Keras.",
            "status_code": 500,
            "path": str(MODEL_PATH),
            "details": _short_exception(error),
        }

    cached["path"] = str(MODEL_PATH)
    cached["mtime"] = current_mtime
    cached["model"] = model
    return model, None


def _extract_model_input_settings(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not isinstance(input_shape, tuple) or len(input_shape) < 4:
        return 224, 224, "rgb"

    height = input_shape[1] or 224
    width = input_shape[2] or 224
    channels = input_shape[3] if len(input_shape) > 3 else 3
    color_mode = "grayscale" if channels == 1 else "rgb"
    return int(height), int(width), color_mode


def _predict_with_model(model, image_paths):
    target_h, target_w, color_mode = _extract_model_input_settings(model)
    samples = []

    for image_path in image_paths:
        try:
            image = tf.keras.utils.load_img(image_path, target_size=(target_h, target_w), color_mode=color_mode)
            image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
            predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
        except Exception:
            continue

        pred_array = np.asarray(predictions)
        if pred_array.ndim == 1:
            pred_array = np.expand_dims(pred_array, axis=0)

        if pred_array.shape[-1] <= 1:
            prob = float(pred_array.reshape(-1)[0])
            class_index = 1 if prob >= 0.5 else 0
            confidence = prob if class_index == 1 else 1 - prob
        else:
            class_index = int(np.argmax(pred_array[0]))
            confidence = float(np.max(pred_array[0]))

        actual_label = image_path.stem.split("_")[0].capitalize()
        static_relative = image_path.relative_to(STATIC_DIR).as_posix()
        samples.append(
            {
                "image_url": f"/static/{static_relative}",
                "predicted_label": f"Class {class_index}",
                "confidence": round(confidence, 4),
                "actual_label": actual_label,
                "notes": "Predicted with models/model.keras",
            }
        )

    return samples


def _extract_upload_bytes(upload):
    if upload is None:
        return None, None, {
            "type": "missing_upload_field",
            "message": "No image file was submitted. Use multipart field name 'image'.",
            "status_code": 400,
        }

    filename = (upload.filename or "").strip()
    if not filename:
        return None, None, {
            "type": "missing_upload_filename",
            "message": "Uploaded file is missing a filename.",
            "status_code": 400,
        }

    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_IMAGE_SUFFIXES:
        return None, None, {
            "type": "invalid_file_type",
            "message": "Only JPEG uploads are allowed (.jpg or .jpeg).",
            "status_code": 400,
            "details": f"Received extension: {suffix or '(none)'}",
        }

    image_bytes = upload.read()
    if not image_bytes:
        return None, None, {
            "type": "empty_upload",
            "message": "Uploaded image is empty.",
            "status_code": 400,
        }

    return image_bytes, filename, None


def _predict_uploaded_image(model, image_bytes):
    target_h, target_w, color_mode = _extract_model_input_settings(model)

    try:
        image = tf.keras.utils.load_img(
            io.BytesIO(image_bytes),
            target_size=(target_h, target_w),
            color_mode=color_mode,
        )
        image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    except Exception as error:
        return None, {
            "type": "invalid_image_content",
            "message": "Uploaded JPEG could not be decoded into an image tensor.",
            "status_code": 400,
            "details": _short_exception(error),
        }

    try:
        predictions = model.predict(np.expand_dims(image_array, axis=0), verbose=0)
    except Exception as error:
        return None, {
            "type": "prediction_failure",
            "message": "Model loaded, but inference failed for the uploaded image.",
            "status_code": 500,
            "details": _short_exception(error),
        }

    pred_array = np.asarray(predictions)
    if pred_array.ndim == 1:
        pred_array = np.expand_dims(pred_array, axis=0)

    if pred_array.shape[-1] <= 1:
        prob = float(pred_array.reshape(-1)[0])
        class_index = 1 if prob >= 0.5 else 0
        confidence = prob if class_index == 1 else 1 - prob
    else:
        class_index = int(np.argmax(pred_array[0]))
        confidence = float(np.max(pred_array[0]))

    return {
        "predicted_label": f"Class {class_index}",
        "class_index": class_index,
        "confidence": round(confidence, 4),
        "model_file_name": MODEL_PATH.name,
    }, None


@app.context_processor
def inject_layout_context():
    # Keep nav config in one place so adding routes is beginner-friendly.
    return {
        "nav_items": _build_nav_items(),
        "current_endpoint": request.endpoint or "",
    }


@app.errorhandler(RequestEntityTooLarge)
def handle_payload_too_large(error):
    payload, status_code = _build_api_error(
        "payload_too_large",
        "Uploaded file exceeds the 5 MB limit.",
        413,
        details=_short_exception(error),
    )
    payload["data"] = {}
    return jsonify(payload), status_code


@app.route("/")
def index():
    return render_template(
        "index.html",
        main_title="Animal Classifier",
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        main_title="About Page",
    )


@app.route("/api/model-stats")
def get_model_stats():
    stats, error = _load_stats_from_history()
    if stats:
        return jsonify(stats)
    return _json_error_response(error, {})


@app.route("/api/model-history")
def get_model_history():
    history, _, error = _load_history_dict()
    if history:
        return jsonify(history)
    return _json_error_response(error, {})


@app.route("/api/sample-results")
def get_sample_results():
    model, error = _get_cached_model()
    if model is None:
        return _json_error_response(error, [])

    image_paths = _list_sample_images()
    if not image_paths:
        error = {
            "type": "missing_sample_images",
            "message": "No supported sample images were found.",
            "status_code": 404,
            "path": "; ".join(str(path) for path in SAMPLE_IMAGE_DIR_CANDIDATES),
            "details": f"Expected one of: {', '.join(sorted(SAMPLE_IMAGE_SUFFIXES))}",
        }
        return _json_error_response(error, [])

    predictions = _predict_with_model(model, image_paths)
    if predictions:
        return jsonify(predictions)

    error = {
        "type": "prediction_failure",
        "message": "Model loaded, but no predictions could be generated from the available images.",
        "status_code": 500,
    }
    return _json_error_response(error, [])


@app.route("/api/predict-image", methods=["POST"])
def predict_image():
    image_bytes, filename, error = _extract_upload_bytes(request.files.get("image"))
    if error:
        return _json_error_response(error, {})

    model, error = _get_cached_model()
    if model is None:
        return _json_error_response(error, {})

    prediction, error = _predict_uploaded_image(model, image_bytes)
    if error:
        return _json_error_response(error, {})

    prediction["uploaded_filename"] = filename
    return jsonify(prediction)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
