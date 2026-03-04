import os
import tempfile
import unittest

import numpy as np

from func import ImageNormalizer, LabelEncoder, keras_ds_train_test_split, clean_rawimg


class TestFunc(unittest.TestCase):
    def test_image_normalizer_scales_to_unit_interval(self):
        x = np.array([[[[0.0, 127.5, 255.0]]]], dtype=np.float32)
        normalizer = ImageNormalizer().fit(x)
        out = normalizer.transform(x)

        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out.min()), 0.0)
        self.assertAlmostEqual(float(out.max()), 1.0)

    def test_label_encoder_one_hot_shape(self):
        y = np.array([0, 2, 1], dtype=np.int32)
        encoded = LabelEncoder(num_classes=4).fit(y).transform(y)

        self.assertEqual(encoded.shape, (3, 4))
        self.assertTrue(np.allclose(encoded.sum(axis=1), 1.0))

    def test_split_raises_for_invalid_or_single_class_path(self):
        with self.assertRaises(ValueError):
            keras_ds_train_test_split(None, seed=40, path="this/path/does/not/exist")

        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "single_class"), exist_ok=True)
            with self.assertRaises(ValueError):
                keras_ds_train_test_split(None, seed=40, path=tmp_dir)

    def test_clean_rawimg_removes_non_jfif(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder_name = "test_class"
            os.makedirs(os.path.join(tmp_dir, folder_name))

            # Create a dummy valid JFIF file (must contain b"JFIF" in first 10 bytes)
            valid_path = os.path.join(tmp_dir, folder_name, "valid.jpg")
            with open(valid_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

            # Create a dummy invalid file
            invalid_path = os.path.join(tmp_dir, folder_name, "invalid.txt")
            with open(invalid_path, "wb") as f:
                f.write(b"not_an_image")

            clean_rawimg([folder_name], tmp_dir)

            self.assertTrue(os.path.exists(valid_path), "Valid JFIF file should remain.")
            self.assertFalse(os.path.exists(invalid_path), "Non-JFIF file should be removed.")


if __name__ == "__main__":
    unittest.main()
