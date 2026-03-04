import io
import tempfile
import unittest
from pathlib import Path

import app as app_module


class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_pages_return_200(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/about").status_code, 200)

    def test_model_stats_api_returns_object(self):
        response = self.client.get("/api/model-stats")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertIn("model_file_status", payload)

    def test_model_history_api_returns_series_object(self):
        response = self.client.get("/api/model-history")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload, "Expected at least one history metric.")
        first_key = next(iter(payload))
        self.assertIsInstance(payload[first_key], list)

    def test_sample_results_api_returns_controlled_response(self):
        response = self.client.get("/api/sample-results")
        payload = response.get_json()

        if response.status_code == 200:
            self.assertIsInstance(payload, list)
        else:
            self.assertIn(response.status_code, (404, 500))
            self.assertIn("error", payload)
            self.assertIsInstance(payload["error"], dict)
            self.assertIn("type", payload["error"])
            self.assertIn("message", payload["error"])

    def test_model_stats_missing_history_returns_typed_error(self):
        original_histories_dir = app_module.HISTORIES_DIR
        original_model_path = app_module.MODEL_PATH

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                app_module.HISTORIES_DIR = tmp_path / "histories"
                app_module.MODEL_PATH = tmp_path / "models" / "model.keras"
                app_module.HISTORIES_DIR.mkdir(parents=True, exist_ok=True)

                response = self.client.get("/api/model-stats")
                payload = response.get_json()

                self.assertEqual(response.status_code, 404)
                self.assertEqual(payload["error"]["type"], "missing_history_file")
                self.assertIn("data", payload)
        finally:
            app_module.HISTORIES_DIR = original_histories_dir
            app_module.MODEL_PATH = original_model_path

    def test_model_history_missing_history_returns_typed_error(self):
        original_histories_dir = app_module.HISTORIES_DIR

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                app_module.HISTORIES_DIR = tmp_path / "histories"
                app_module.HISTORIES_DIR.mkdir(parents=True, exist_ok=True)

                response = self.client.get("/api/model-history")
                payload = response.get_json()

                self.assertEqual(response.status_code, 404)
                self.assertEqual(payload["error"]["type"], "missing_history_file")
                self.assertIn("data", payload)
        finally:
            app_module.HISTORIES_DIR = original_histories_dir

    def test_predict_image_missing_field_returns_typed_error(self):
        response = self.client.post("/api/predict-image", data={})
        payload = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "missing_upload_field")
        self.assertIn("data", payload)

    def test_predict_image_wrong_extension_returns_typed_error(self):
        response = self.client.post(
            "/api/predict-image",
            data={"image": (io.BytesIO(b"fake"), "sample.txt")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_file_type")
        self.assertIn("data", payload)

    def test_predict_image_model_load_failure_returns_typed_error(self):
        original_model_path = app_module.MODEL_PATH
        original_model_cache = dict(app_module.MODEL_CACHE)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                bad_model_path = tmp_path / "models" / "model.keras"
                bad_model_path.parent.mkdir(parents=True, exist_ok=True)
                bad_model_path.write_bytes(b"not a valid keras model archive")

                app_module.MODEL_PATH = bad_model_path
                app_module.MODEL_CACHE = {"path": None, "mtime": None, "model": None}

                response = self.client.post(
                    "/api/predict-image",
                    data={"image": (io.BytesIO(b"not-empty"), "sample.jpg")},
                    content_type="multipart/form-data",
                )
                payload = response.get_json()

                self.assertEqual(response.status_code, 500)
                self.assertIn(
                    payload["error"]["type"],
                    {"model_file_read_error", "model_deserialization_error"},
                )
                self.assertIn("data", payload)
        finally:
            app_module.MODEL_PATH = original_model_path
            app_module.MODEL_CACHE = original_model_cache


if __name__ == "__main__":
    unittest.main()
