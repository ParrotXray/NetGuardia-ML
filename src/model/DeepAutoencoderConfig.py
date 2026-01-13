from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepAutoencoderConfig:
    clip_min: float = -5.0
    clip_max: float = 5.0
    winsorize_lower: float = 0.005
    winsorize_upper: float = 0.995
    fill_value: float = 0.0

    encoding_dim: int = 16
    layer_sizes: List[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64])
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.3, 0.25, 0.2, 0.15, 0.0]
    )
    l2_reg: float = 0.0001
    learning_rate: float = 0.001
    clipnorm: float = 1.0
    batch_size: int = 1024
    epochs: int = 100
    validation_split: float = 0.15
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    rf_min_samples_split: int = 10
    rf_min_samples_leaf: int = 5
    rf_max_features: str = "sqrt"
    rf_n_jobs: int = -1
    rf_random_state: int = 42
    rf_train_samples: int = 50000

    ensemble_strategies: List[float] = field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7]
    )

    output_csv_name: str = "output_deep_ae_ensemble.csv"
    output_model_ae: str = "deep_autoencoder.keras"
    output_model_rf: str = "random_forest.pkl"
    output_config: str = "deep_ae_ensemble_config.pkl"
    output_plot: str = "deep_ae_ensemble_analysis.png"
