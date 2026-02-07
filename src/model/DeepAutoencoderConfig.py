"""
Deep Autoencoder Configuration Module.

This module defines the configuration dataclass for training a Deep Autoencoder
combined with Isolation Forest for network traffic anomaly detection.
The autoencoder learns compressed latent representations of normal traffic patterns,
while the Isolation Forest performs anomaly detection in the latent space.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepAutoencoderConfig:
    """
    Configuration for Deep Autoencoder with Isolation Forest (Latent Space).

    Architecture: AE Encoder (feature extraction) -> IF (anomaly detection in latent space)

    The autoencoder compresses input features into a low-dimensional latent space.
    The Isolation Forest then operates on these latent features for anomaly detection,
    following the literature-standard approach.

    Attributes:
        Data Preprocessing:
            clip_min: Minimum value for clipping normalized features.
            clip_max: Maximum value for clipping normalized features.
            winsorize_lower: Lower percentile for winsorization to handle outliers.
            winsorize_upper: Upper percentile for winsorization to handle outliers.
            fill_value: Value used to fill missing or invalid entries.

        Autoencoder Architecture:
            encoding_dim: Dimension of the bottleneck (latent) layer.
            layer_sizes: List of hidden layer sizes for encoder (decoder mirrors this).
            dropout_rates: Dropout rate for each layer to prevent overfitting.
            l2_reg: L2 regularization factor for weight decay.

        Autoencoder Training:
            learning_rate: Initial learning rate for Adam optimizer.
            clipnorm: Gradient clipping norm to prevent exploding gradients.
            batch_size: Number of samples per training batch.
            epochs: Maximum number of training epochs.
            validation_split: Fraction of data used for validation.
            early_stopping_patience: Epochs to wait before early stopping.
            reduce_lr_patience: Epochs to wait before reducing learning rate.
            reduce_lr_factor: Factor to reduce learning rate by.
            min_lr: Minimum learning rate threshold.

        Isolation Forest Parameters:
            if_n_estimators: Number of isolation trees.
            if_contamination: Expected proportion of anomalies in training data.
            if_max_samples: Number of samples to draw for training each tree.
            if_max_features: Number of features to draw for each tree.
            if_random_state: Random seed for reproducibility.

        Evaluation Configuration:
            percentiles: Percentile thresholds for anomaly score evaluation.
    """

    # Data Preprocessing Parameters
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    # Autoencoder Architecture Parameters
    encoding_dim: int = 8
    layer_sizes: List[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.0]
    )
    l2_reg: float = 0.0001

    # Autoencoder Training Parameters
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 1024
    epochs: int = 350
    validation_split: float = 0.15
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    # Isolation Forest Parameters
    if_n_estimators: int = 100
    if_contamination: float = 0.05
    if_max_samples: str = "auto"
    if_max_features: float = 1.0
    if_random_state: int = 42

    # Anomaly Score Thresholds
    percentiles: List[float] = field(
        default_factory=lambda: [97.0, 98.0, 99.0, 99.5, 99.7, 99.9]
    )

    # Confidence Thresholds
    min_precision: float = 0.6
    min_tpr: float = 0.80
