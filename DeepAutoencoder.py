import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from Logger import Logger


@dataclass
class DeepAutoencoderConfig: ...


class DeepAutoencoder:
    def __init__(self) -> None:
        self.labels: Optional[pd.Series] = None
        self.log: Logger = Logger("DeepAutoencoder")
        self.datasets: List[pd.DataFrame] = []

    def tensorflow_version(self) -> None:
        self.log.info(f"TensorFlow: {tf.__version__}")
        self.log.info(f"GPU: {tf.config.list_physical_devices('GPU')}")

    def load_dataset(self, file: str) -> None:
        try:
            self.log.info(f"Loading {file}")
            df = pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
            df.columns = df.columns.str.strip()
            self.labels = df["Label"].copy()
            self.datasets.append(df)

            self.log.info(
                f"Total Sample: {len(df):,}, "
                f"BENIGN: {(self.labels == 'BENIGN').sum():,}, "
                f"Attack: {(self.labels != 'BENIGN').sum():,}"
            )

        except FileNotFoundError:
            self.log.warning(f"File does not exist: {file}")
        except Exception as e:
            self.log.error(f"Error: {e}")

    def load_datasets(self, csv_dir: str) -> None:
        self.log.info(f"Loading datasets from {csv_dir}...")

        csv_files = list(Path(csv_dir).glob("*.csv"))
        if not csv_files:
            self.log.warning(f"No CSV files found in {csv_dir}")
            return

        for file_path in csv_files:
            self.load_dataset(str(file_path))

        if not self.datasets:
            raise ValueError("No dataset was successfully loaded!")

        self.log.info(f"Successfully loaded {len(self.datasets)} datasets")
