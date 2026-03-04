from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import keras
from keras.utils import to_categorical
from sklearn.pipeline import Pipeline
import os

# ---
class ImageNormalizer(BaseEstimator, TransformerMixin):
    """Normalize images to float32 in [0, 1] range."""
    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        return X.astype('float32') / 255.0
class ImageResizer(BaseEstimator, TransformerMixin):
    """Resize images to specified dimensions using bilinear interpolation"""
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        from scipy.ndimage import zoom
        h, w = self.target_size
        scale_h = h / X.shape[1]
        scale_w = w / X.shape[2]
        return zoom(X, (1, scale_h, scale_w, 1), order=1)
class LabelEncoder(BaseEstimator, TransformerMixin):
    """Convert labels to one-hot encoded form"""
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, y):
        return to_categorical(y, self.num_classes)
def preprocess_data(x_train, y_train, x_test, y_test) -> tuple:
    """
    Preprocess CIFAR-10 data with normalization, resizing, and one-hot encoding

    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels

    Returns:
        Tuple of (x_train_processed, y_train_processed, x_test_processed, y_test_processed)
    """
    # Image preprocessing pipeline
    image_pipeline = Pipeline([
        ('normalizer', ImageNormalizer()),
        ('resizer', ImageResizer(target_size=(64, 64))),
    ])

    # Label preprocessing pipeline
    label_pipeline = Pipeline([
        ('encoder', LabelEncoder(num_classes=10)),
    ])

    # Apply transformations
    x_train_processed = image_pipeline.fit_transform(x_train)
    x_test_processed = image_pipeline.transform(x_test)

    y_train_processed = label_pipeline.fit_transform(y_train)
    y_test_processed = label_pipeline.transform(y_test)

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed

def keras_ds_train_test_split(keras_dataset, seed:int=None, path:str='') -> tuple:
    '''
    Split a Keras image dataset into training and test sets.
    Args:
        keras_dataset: A Keras image dataset created using image_dataset_from_directory.
        seed: An integer random seed for reproducibility.
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    '''
    dataset_dir = path if path else (keras_dataset if isinstance(keras_dataset, str) else '')
    if not dataset_dir:
        dataset_dir = 'raw-img'
    dataset_dir = os.path.abspath(dataset_dir)

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    raw_img_candidate = os.path.join(dataset_dir, 'raw-img')
    if os.path.isdir(raw_img_candidate):
        raw_img_classes = [
            entry for entry in os.listdir(raw_img_candidate)
            if os.path.isdir(os.path.join(raw_img_candidate, entry))
        ]
        if len(raw_img_classes) >= 2:
            dataset_dir = raw_img_candidate

    class_dirs = [
        entry for entry in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, entry))
    ]
    if len(class_dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class folders in '{dataset_dir}', found {len(class_dirs)}."
        )

    train_dataset = keras.utils.image_dataset_from_directory(
        dataset_dir, label_mode='int', labels='inferred',
        image_size=(64, 64), shuffle=True, seed=seed,
        batch_size=32, color_mode='rgb',
        validation_split=0.2, subset='training')

    test_dataset = keras.utils.image_dataset_from_directory(
        dataset_dir, label_mode='int', labels='inferred',
        image_size=(64, 64), shuffle=True, seed=seed,
        batch_size=32, color_mode='rgb',
        validation_split=0.2, subset='validation')

    if hasattr(train_dataset, 'file_paths') and hasattr(test_dataset, 'file_paths'):
        overlap = set(train_dataset.file_paths).intersection(test_dataset.file_paths)
        if overlap:
            raise ValueError(
                f"Data leakage detected: {len(overlap)} overlapping files between train and validation."
            )

    x_train, y_train = [], []
    for images, labels in train_dataset:
        x_train.append(images.numpy())
        y_train.append(labels.numpy())
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = [], []
    for images, labels in test_dataset:
        x_test.append(images.numpy())
        y_test.append(labels.numpy())
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    return x_train, y_train, x_test, y_test

def clean_rawimg(folders:list, dataset_path:str) -> None:
    """
    Clean the raw image dataset by removing non-JFIF images.

    Iterates through specified folders within the dataset path and removes files
    that do not contain the JFIF header.

    Args:
        folders: List of folder names to process.
        dataset_path: Base directory path containing the folders.
    """

    num_skipped = 0
    for folder_name in folders:
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_name}, directory not found at {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "rb") as fobj:
                    # Check if it's a valid JFIF/JPEG
                    is_jfif = b"JFIF" in fobj.peek(10)

                if not is_jfif:
                    num_skipped += 1
                    os.remove(fpath) # delete files
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
    print(f"Identified {num_skipped} non-JFIF images.")

def download_dataset(path:str, seed:int=40):
    import kagglehub
    import shutil

    if 'COLAB_GPU' in os.environ: # if working is google colab
        print('Working in COLAB')
        # Download latest version
        path = kagglehub.dataset_download("alessiocorrado99/animals10")

        dataset_path = path
        raw_img_candidate = os.path.join(path, 'raw-img')
        if os.path.isdir(raw_img_candidate):
            dataset_path = raw_img_candidate
    else:
        path = kagglehub.dataset_download("alessiocorrado99/animals10")
        destination = os.path.join(os.path.curdir,'raw-img')
        os.makedirs(destination, exist_ok=True)

        for item in os.listdir(path):
            src = os.path.join(path, item)
            dst = os.path.join(destination, item)
            shutil.move(src, dst)
        print('Working locally')
        dataset_path = os.path.join(path, 'raw-img')

    print("Path to dataset files:", dataset_path)
    dataset = keras.utils.image_dataset_from_directory(
        dataset_path, label_mode='int', labels='inferred',
        image_size=(64, 64), shuffle=False, seed=seed,
        batch_size=32, color_mode='rgb',
    )

    return dataset, dataset_path
# ---
