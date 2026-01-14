from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class MLPConfig:
    test_size: float = 0.2
    random_state: int = 42
    fill_value: float = 0.0

    smote_ratio: float = 0.5
    smote_k_neighbors: int = 5

    layer_sizes: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dropout_rates: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])

    learning_rate: float = 0.001
    batch_size: int = 512
    epochs: int = 100
    validation_split: float = 0.0
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    clip_min: float = -5.0
    clip_max: float = 5.0

    # output_model_name: str = "mlp_improved"
    # output_encoder_name: str = "label_encoder_improved"
    # output_config_name: str = "mlp_improved_config"
    # output_plot_name: str = "mlp_improved_analysis"
    # output_csv_name: str = "output_mlp_improved"
