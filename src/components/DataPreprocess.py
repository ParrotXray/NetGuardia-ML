import os

import pandas as pd
import numpy as np
from pathlib import Path
import ujson
from typing import List, Optional, Dict, Any
from utils import Logger
from model import PreprocessConfig


class DataPreprocess:
    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:

        self.datasets: List[pd.DataFrame] = []
        self.combined_data: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.labels: Optional[pd.Series] = None

        self.config: Optional[PreprocessConfig] = config or PreprocessConfig()

        self.log: Logger = Logger("DataPreprocess")

    def load_dataset(self, file: str) -> None:
        try:
            self.log.info(f"Loading {file}")
            df = pd.read_csv(file, encoding="utf-8", encoding_errors="replace")
            df.columns = df.columns.str.strip()
            self.datasets.append(df)
            self.log.info(f"Shape: {df.shape}, Label: {df['Label'].nunique()} Class")
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

    def merge_dataset(self) -> None:
        if not self.datasets:
            raise ValueError("No datasets to merge!")

        self.log.info(f"Merge dataset...")
        self.combined_data = pd.concat(self.datasets, ignore_index=True)

        self.labels = (
            self.combined_data["Label"].str.replace("�", "-", regex=False).copy()
        )

        web_attack_mapping = {
            "Web Attack - Brute Force": "Web Attack",
            "Web Attack - Sql Injection": "Web Attack",
            "Web Attack - XSS": "Web Attack",
            "Infiltration": "Web Attack",
            "Heartbleed": "DoS Slowhttptest",
            "Brute Force -Web": "Web Attack",
            "Brute Force -XSS": "Web Attack",
            "SQL Injection": "Web Attack",
        }
        self.labels = self.labels.replace(web_attack_mapping)

        self.log.info(f"Combined data: {self.combined_data.shape}")

        print("Tag distribution:")
        print(self.labels.value_counts())

    def feature_preparation(self) -> None:
        if self.combined_data is None:
            raise ValueError("No combined data available. Call merge_dataset() first!")

        self.log.info("Feature preparation...")

        selected_features = [
            # === Tier 1: 最重要 (前 11 名) ===
            'Dst Port',                      # 1. 攻擊目標識別
            'Protocol',                      # 2. 協議類型 ⭐ NEW!
            'Flow Duration',                 # 3. 異常連線時間
            'Tot Fwd Pkts',                  # 4. 流量大小
            'Tot Bwd Pkts',                  # 5. 回應模式
            'TotLen Fwd Pkts',               # 6. 數據量
            'TotLen Bwd Pkts',               # 7. 回應數據量
            'Flow Byts/s',                   # 8. 流量速率
            'Flow Pkts/s',                   # 9. 封包速率
            'Init Fwd Win Byts',             # 10. TCP 特徵
            'Init Bwd Win Byts',             # 11. TCP 回應

            # === Tier 2: 重要 (第 12-21 名) ===
            'Fwd Pkt Len Mean',              # 12. 封包大小模式
            'Bwd Pkt Len Mean',              # 13. 回應封包模式
            'Flow IAT Mean',                 # 14. 封包間隔
            'Fwd IAT Mean',                  # 15. 發送節奏
            'Bwd IAT Mean',                  # 16. 回應節奏
            'PSH Flag Cnt',                  # 17. 數據推送
            'ACK Flag Cnt',                  # 18. 確認封包
            'SYN Flag Cnt',                  # 19. 連線建立
            'FIN Flag Cnt',                  # 20. 連線結束
            'RST Flag Cnt',                  # 21. 連線重置

            # === Tier 3: 次要重要 (第 22-27 名) ===
            'Pkt Len Mean',                  # 22. 整體封包大小
            'Pkt Len Std',                   # 23. 封包大小變異
            'Fwd Pkt Len Std',               # 24. 發送變異
            'Bwd Pkt Len Std',               # 25. 回應變異
            'Fwd Seg Size Min',              # 26. 最小段大小
            'Fwd Act Data Pkts',             # 27. 實際數據封包數
        ]

        # 2017
        # selected_features = [
        #     'Destination Port',  # 1. 攻擊目標識別
        #     'Flow Duration',  # 2. 異常連線時間
        #     'Total Fwd Packets',  # 3. 流量大小
        #     'Total Backward Packets',  # 4. 回應模式
        #     'Total Length of Fwd Packets',  # 5. 數據量
        #     'Total Length of Bwd Packets',  # 6. 回應數據量
        #     'Flow Bytes/s',  # 7. 流量速率
        #     'Flow Packets/s',  # 8. 封包速率
        #     'Init_Win_bytes_forward',  # 9. TCP 特徵
        #     'Init_Win_bytes_backward',  # 10. TCP 回應
        #
        #     # === Tier 2: 重要 (第 11-20 名) ===
        #     'Fwd Packet Length Mean',  # 11. 封包大小模式
        #     'Bwd Packet Length Mean',  # 12. 回應封包模式
        #     'Flow IAT Mean',  # 13. 封包間隔
        #     'Fwd IAT Mean',  # 14. 發送節奏
        #     'Bwd IAT Mean',  # 15. 回應節奏
        #     'PSH Flag Count',  # 16. 數據推送
        #     'ACK Flag Count',  # 17. 確認封包
        #     'SYN Flag Count',  # 18. 連線建立
        #     'FIN Flag Count',  # 19. 連線結束
        #     'RST Flag Count',  # 20. 連線重置
        #
        #     # === Tier 3: 次要重要 (第 21-26 名) ===
        #     'Packet Length Mean',  # 21. 整體封包大小
        #     'Packet Length Std',  # 22. 封包大小變異
        #     'Fwd Packet Length Std',  # 23. 發送變異
        #     'Bwd Packet Length Std',  # 24. 回應變異
        #     'min_seg_size_forward',  # 25. 最小段大小
        #     'act_data_pkt_fwd',  # 26. 實際數據封包數
        # ]

        available_features = [
            f for f in selected_features if f in self.combined_data.columns
        ]
        missing_features = set(selected_features) - set(available_features)

        if missing_features:
            self.log.warning(f"Missing features: {missing_features}")

        self.feature_matrix = self.combined_data[available_features].copy()
        self.log.info(f"Selected {len(available_features)} features")

        self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], np.nan)
        self.feature_matrix = self.feature_matrix.fillna(self.config.fill_value)
        self.feature_matrix = self.feature_matrix.clip(
            self.config.clip_min, self.config.clip_max
        )
        self.log.info(f"Feature matrix shape: {self.feature_matrix.shape}")

    def output_result(self) -> None:
        if self.feature_matrix is None:
            raise ValueError(
                "No feature matrix available. Call feature_preparation() first!"
            )

        self.log.info("Saving processed data...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output: pd.DataFrame = self.feature_matrix.copy()
        output["Label"] = self.labels.values

        invalid_labels = ["Unknown", "0", "", "nan"]
        benign_mask = (output["Label"] == "BENIGN") | (output["Label"] == "Benign")
        attack_mask = (output["Label"] != "BENIGN") | (output["Label"] != "Benign") & (
            ~output["Label"].isin(invalid_labels)
        )
        output_benign = output[benign_mask]
        output_attack = output[attack_mask]

        dropped_count = len(output) - len(output_benign) - len(output_attack)
        if dropped_count > 0:
            self.log.warning(f"Dropped {dropped_count:,} rows with invalid labels")

        benign_path = Path("outputs") / "preprocessing_benign.csv"
        attack_path = Path("outputs") / "preprocessing_attack.csv"

        output_benign.to_csv(benign_path, index=False)
        output_attack.to_csv(attack_path, index=False)

        self.log.info(f"BENIGN samples: {len(output_benign):,} -> {benign_path}")
        self.log.info(f"Attack samples: {len(output_attack):,} -> {attack_path}")

        stats = {
            "total_samples": len(self.combined_data),
            "total_features": self.feature_matrix.shape[1],
            "benign_samples": len(output_benign),
            "attack_samples": len(output_attack),
            "label_distribution": self.labels.value_counts().to_dict(),
        }

        stats_path = Path("metadata") / "preprocessing_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            ujson.dump(stats, f, indent=2, ensure_ascii=False)
        self.log.info(f"Statistics save: {stats_path}")