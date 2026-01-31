from dataclasses import dataclass, field
from typing import List


@dataclass
class PreprocessConfig:
    contamination_rate: float = 0.05
    random_state: int = 42
    n_jobs: int = -1
    if_verbose: int = 1

    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0

    selected_features: List[str] = field(
        default_factory=lambda: [
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
    )

    # output_csv_name: str = "output_anomaly"
    # output_stats_name: str = "preprocessing_stats"
    # output_model_name: str = "isolation_forest_model"
