import pandas as pd
import tensorflow as tf
from typing import List, Optional, Dict
from pathlib import Path
from src.utils.Logger import Logger
from src.model.DeepAutoencoderConfig import  DeepAutoencoderConfig
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt

class DeepAutoencoder:
    def __init__(self, config: Optional[DeepAutoencoderConfig] = None):
        self.raw_data: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.features: Optional[pd.DataFrame] = None
        self.binary_labels: Optional[pd.Series] = None

        self.benign_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.test_labels: Optional[pd.Series] = None

        self.scaler: Optional[StandardScaler] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.benign_features_scaled: Optional[np.ndarray] = None
        self.test_features_scaled: Optional[np.ndarray] = None

        self.config: Optional[DeepAutoencoderConfig] = config or DeepAutoencoderConfig()

        self.log: Logger = Logger("DeepAutoencoder")

    def check_tensorflow(self) -> None:
        self.log.info(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        self.log.info(f"GPU: {gpus if gpus else 'No GPU detected'}")

    def load_data(self, file_path: str) -> None:
        self.log.info(f"Loading data from {file_path}...")
        self.raw_data = pd.read_csv(file_path)
        self.raw_data.columns = self.raw_data.columns.str.strip()
        self.labels = self.raw_data["Label"].copy()

        benign_count = (self.labels == 'BENIGN').sum()
        attack_count = (self.labels != 'BENIGN').sum()

        self.log.info(f"Total samples: {len(self.raw_data):,}")
        self.log.info(f"BENIGN: {benign_count:,}")
        self.log.info(f"Attack: {attack_count:,}")

    def prepare_data(self) -> None:
        self.log.info("Preparing data...")

        exclude_cols = ["Label", "anomaly_if"]
        self.features = self.raw_data.drop(columns=exclude_cols, errors="ignore")
        self.features = self.features.select_dtypes(include=[np.number])

        self.binary_labels = (self.labels != "BENIGN").astype(int)

        self.benign_features = self.features[self.binary_labels == 0].copy()
        self.test_features = self.features.copy()
        self.test_labels = self.binary_labels.copy()

        self.log.info(f"BENIGN training samples: {len(self.benign_features):,}")
        self.log.info(f"Total test samples: {len(self.test_features):,}")
        self.log.info(f"Number of features: {self.features.shape[1]}")

    def preprocess_data(self) -> None:
        self.log.info("Preprocessing data...")

        self.benign_features = self.benign_features.replace(
            [np.inf, -np.inf], np.nan).fillna(self.config.fill_value)
        self.test_features = self.test_features.replace(
            [np.inf, -np.inf], np.nan).fillna(self.config.fill_value)

        self.clip_params = {}
        for col in self.benign_features.columns:
            lower = self.benign_features[col].quantile(self.config.winsorize_lower)
            upper = self.benign_features[col].quantile(self.config.winsorize_upper)
            self.benign_features[col] = np.clip(self.benign_features[col], lower, upper)
            self.test_features[col] = np.clip(self.test_features[col], lower, upper)
            self.clip_params[col] = {"lower": float(lower), "upper": float(upper)}

        self.scaler = StandardScaler()
        self.benign_features_scaled = self.scaler.fit_transform(self.benign_features)
        self.test_features_scaled = self.scaler.transform(self.test_features)

        self.benign_features_scaled = np.clip(
            self.benign_features_scaled,
            self.config.clip_min,
            self.config.clip_max
        )
        self.test_features_scaled = np.clip(
            self.test_features_scaled,
            self.config.clip_min,
            self.config.clip_max
        )

        self.log.info("Preprocessing completed")