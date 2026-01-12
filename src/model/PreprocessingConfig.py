from dataclasses import dataclass

@dataclass
class PreprocessingConfig:
    contamination_rate: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    if_verbose: int = 1

    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

    output_csv_name: str = "output_anomaly.csv"
    output_stats_name: str = "preprocessing_stats.json"
    output_model_name: str = "isolation_forest_model.joblib"