import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import CSVLogger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import DeepAutoencoderConfig
from utils import Logger


class AutoencoderModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        encoding_dim: int,
        dropout_rates: List[float],
        l2_reg: float = 0.0001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        encoder_layers = []
        prev_size = input_dim
        for i, (size, dropout) in enumerate(zip(layer_sizes, dropout_rates)):
            encoder_layers.append(nn.Linear(prev_size, size))
            encoder_layers.append(nn.BatchNorm1d(size))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        encoder_layers.append(nn.Linear(prev_size, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_size = encoding_dim
        for i, (size, dropout) in enumerate(
            zip(reversed(layer_sizes), reversed(dropout_rates))
        ):
            decoder_layers.append(nn.Linear(prev_size, size))
            decoder_layers.append(nn.BatchNorm1d(size))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_size = size

        decoder_layers.append(nn.Linear(prev_size, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.l2_reg = l2_reg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderLightningModule(L.LightningModule):
    def __init__(
        self,
        model: AutoencoderModel,
        learning_rate: float = 0.001,
        clipnorm: float = 1.0,
        reduce_lr_factor: float = 0.5,
        reduce_lr_patience: int = 8,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.min_lr = min_lr
        self.loss_fn = nn.MSELoss()

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        mae = torch.mean(torch.abs(x_hat - x))

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.model.l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class DeepAutoencoder:
    def __init__(self, config: Optional[DeepAutoencoderConfig] = None) -> None:
        self.benign_data: Optional[pd.DataFrame] = None
        self.attack_data: Optional[pd.DataFrame] = None
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

        self.autoencoder_model: Optional[AutoencoderModel] = None
        self.lightning_module: Optional[AutoencoderLightningModule] = None
        self.isolation_forest_model: Optional[IsolationForest] = None

        self.latent_benign: Optional[np.ndarray] = None
        self.latent_test: Optional[np.ndarray] = None

        self.if_predictions: Optional[np.ndarray] = None
        self.if_scores: Optional[np.ndarray] = None

        self.if_metrics: Optional[Dict[str, Any]] = None

        self.training_history: Optional[Dict[str, List[float]]] = None

        self.config: Optional[DeepAutoencoderConfig] = config or DeepAutoencoderConfig()

        self.log: Logger = Logger("DeepAutoencoder")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_tensorflow(self) -> None:
        self.log.info(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            self.log.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.log.info(f"CUDA: {torch.version.cuda}")
        else:
            self.log.info("GPU: No GPU detected, using CPU")

    def load_data(self) -> None:
        self.log.info("Loading data from outputs/preprocessing_benign.csv...")
        self.benign_data = pd.read_csv("./outputs/preprocessing_benign.csv")
        self.benign_data.columns = self.benign_data.columns.str.strip()

        self.log.info("Loading data from outputs/preprocessing_attack.csv...")
        self.attack_data = pd.read_csv("./outputs/preprocessing_attack.csv")
        self.attack_data.columns = self.attack_data.columns.str.strip()

        print(f"BENIGN samples: {len(self.benign_data):,}")
        print(f"Attack samples: {len(self.attack_data):,}")

    def prepare_data(self) -> None:
        self.log.info("Preparing data...")

        exclude_cols = ["Label"]

        self.benign_features = self.benign_data.drop(
            columns=exclude_cols, errors="ignore"
        ).select_dtypes(include=[np.number])

        attack_features = self.attack_data.drop(
            columns=exclude_cols, errors="ignore"
        ).select_dtypes(include=[np.number])

        self.features = pd.concat(
            [self.benign_features, attack_features], ignore_index=True
        )
        self.labels = pd.concat(
            [self.benign_data["Label"], self.attack_data["Label"]], ignore_index=True
        )
        self.binary_labels = (~self.labels.isin(["BENIGN", "Benign"])).astype(int)

        self.test_features = self.features.copy()
        self.test_labels = self.binary_labels.copy()

        print(f"BENIGN training samples: {len(self.benign_features):,}")
        print(f"Total test samples: {len(self.test_features):,}")
        print(f"Number of features: {self.features.shape[1]}")

    def preprocess_data(self) -> None:
        self.log.info("Preprocessing data...")

        self.benign_features = self.benign_features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(self.config.fill_value)
        self.test_features = self.test_features.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(self.config.fill_value)

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
            self.benign_features_scaled, self.config.clip_min, self.config.clip_max
        )
        self.test_features_scaled = np.clip(
            self.test_features_scaled, self.config.clip_min, self.config.clip_max
        )

        self.log.info("Preprocessing completed")

    def build_autoencoder(self) -> None:
        self.log.info("Building Deep Autoencoder...")

        input_dim = self.benign_features_scaled.shape[1]

        layer_info = " -> ".join(
            [str(input_dim)]
            + [str(s) for s in self.config.layer_sizes]
            + [str(self.config.encoding_dim)]
        )
        self.log.info(f"Architecture: {layer_info}")

        self.autoencoder_model = AutoencoderModel(
            input_dim=input_dim,
            layer_sizes=self.config.layer_sizes,
            encoding_dim=self.config.encoding_dim,
            dropout_rates=self.config.dropout_rates,
            l2_reg=self.config.l2_reg,
        )

        self.lightning_module = AutoencoderLightningModule(
            model=self.autoencoder_model,
            learning_rate=self.config.learning_rate,
            clipnorm=self.config.clipnorm,
            reduce_lr_factor=self.config.reduce_lr_factor,
            reduce_lr_patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
        )

        total_params = sum(p.numel() for p in self.autoencoder_model.parameters())
        self.log.info(f"Total parameters: {total_params:,}")

    def train_autoencoder(self) -> None:
        self.log.info("Training Deep Autoencoder with PyTorch Lightning...")

        train_size = int(
            len(self.benign_features_scaled) * (1 - self.config.validation_split)
        )
        indices = np.random.permutation(len(self.benign_features_scaled))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_data = torch.FloatTensor(self.benign_features_scaled[train_indices])
        val_data = torch.FloatTensor(self.benign_features_scaled[val_indices])

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        num_workers = 4 if os.name != "nt" else 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False,
        )

        os.makedirs("./artifacts", exist_ok=True)
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=1e-6,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath="./artifacts",
                filename="autoencoder_temp",
                monitor="val_loss",
                save_top_k=1,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        trainer = L.Trainer(
            max_epochs=self.config.epochs,
            accelerator="auto",
            devices=1,
            callbacks=callbacks,
            enable_progress_bar=True,
            gradient_clip_val=self.config.clipnorm,
            log_every_n_steps=50,
            logger=True,
        )

        trainer.fit(self.lightning_module, train_loader, val_loader)

        best_model_path = callbacks[1].best_model_path
        if best_model_path:
            self.lightning_module = AutoencoderLightningModule.load_from_checkpoint(
                best_model_path,
                model=self.autoencoder_model,
            )
            self.log.info(f"Loaded best model from {best_model_path}")

        self.training_history = {
            "loss": [],
            "val_loss": [],
        }

        epochs_trained = (
            trainer.current_epoch + 1 if trainer.current_epoch else self.config.epochs
        )

        print(f"Training completed: {epochs_trained} epochs")
        print(f"Best validation loss: {callbacks[1].best_model_score:.6f}")

    def extract_latent_features(self) -> None:
        self.log.info("Extracting latent features via AE encoder...")

        self.lightning_module.eval()
        self.lightning_module.to(self.device)

        batch_size = 2048

        with torch.no_grad():
            # Extract latent features for benign (training) data
            latent_benign_parts = []
            for i in range(0, len(self.benign_features_scaled), batch_size):
                batch = torch.FloatTensor(
                    self.benign_features_scaled[i : i + batch_size]
                ).to(self.device)
                latent = self.autoencoder_model.encode(batch)
                latent_benign_parts.append(latent.cpu().numpy())
            self.latent_benign = np.vstack(latent_benign_parts)

            # Extract latent features for test (all) data
            latent_test_parts = []
            for i in range(0, len(self.test_features_scaled), batch_size):
                batch = torch.FloatTensor(
                    self.test_features_scaled[i : i + batch_size]
                ).to(self.device)
                latent = self.autoencoder_model.encode(batch)
                latent_test_parts.append(latent.cpu().numpy())
            self.latent_test = np.vstack(latent_test_parts)

        print(f"Original feature dim: {self.benign_features_scaled.shape[1]}")
        print(f"Latent feature dim: {self.latent_benign.shape[1]}")
        print(f"Benign latent samples: {self.latent_benign.shape[0]:,}")
        print(f"Test latent samples: {self.latent_test.shape[0]:,}")

    def train_isolation_forest(self) -> None:
        self.log.info("Training Isolation Forest on latent space...")

        self.isolation_forest_model = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.if_contamination,
            max_samples=self.config.if_max_samples,
            max_features=self.config.if_max_features,
            random_state=self.config.if_random_state,
            verbose=1,
            n_jobs=-1,
        )

        self.isolation_forest_model.fit(self.latent_benign)
        self.log.info("Isolation Forest training completed on latent space")

        print(f"IF trained on {self.latent_benign.shape[0]:,} benign latent samples")
        print(f"Latent dimensions: {self.latent_benign.shape[1]}")

    def predict_isolation_forest(self) -> None:
        self.log.info("Running Isolation Forest predictions on latent space...")

        # IF predictions: 1 = normal (inlier), -1 = anomaly (outlier)
        self.if_predictions = self.isolation_forest_model.predict(self.latent_test)
        # IF scores: more negative = more anomalous
        self.if_scores = self.isolation_forest_model.score_samples(self.latent_test)

        n_anomalies = (self.if_predictions == -1).sum()
        n_normal = (self.if_predictions == 1).sum()

        print(f"IF predictions: {n_normal:,} normal, {n_anomalies:,} anomalies")
        print(
            f"IF score range: [{self.if_scores.min():.4f}, {self.if_scores.max():.4f}]"
        )
        print(f"IF score mean: {self.if_scores.mean():.4f}")
        print(f"IF score median: {np.median(self.if_scores):.4f}")

    def evaluate_detection(self) -> None:
        self.log.info("Evaluating anomaly detection performance...")

        # IF: -1 = anomaly, 1 = normal
        # Binary: 1 = attack, 0 = benign
        pred_anomaly = (self.if_predictions == -1).astype(int)
        true_attack = self.test_labels.values

        tp = ((true_attack == 1) & (pred_anomaly == 1)).sum()
        fp = ((true_attack == 0) & (pred_anomaly == 1)).sum()
        fn = ((true_attack == 1) & (pred_anomaly == 0)).sum()
        tn = ((true_attack == 0) & (pred_anomaly == 0)).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0

        self.if_metrics = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "precision": float(precision),
            "f1": float(f1),
        }

        print(f"\n{'=' * 60}")
        print(f"AE (Encoder) -> IF (Latent Space) Detection Results")
        print(f"{'=' * 60}")
        print(f"TP: {tp:,}  FP: {fp:,}")
        print(f"FN: {fn:,}  TN: {tn:,}")
        print(f"TPR (Recall): {tpr:.2%}")
        print(f"FPR: {fpr:.2%}")
        print(f"Precision: {precision:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"{'=' * 60}\n")

        # Score-based threshold evaluation
        print("Score-based threshold analysis:")
        header = f"{'Percentile':>12} {'Threshold':>10} {'TPR':>7} {'FPR':>7} {'Prec':>7} {'F1':>7}"
        print(header)
        print("-" * 60)

        benign_scores = self.if_scores[true_attack == 0]
        for percentile in self.config.percentiles:
            threshold = np.percentile(benign_scores, 100 - percentile)
            pred = (self.if_scores < threshold).astype(int)

            tp_t = ((true_attack == 1) & (pred == 1)).sum()
            fp_t = ((true_attack == 0) & (pred == 1)).sum()
            fn_t = ((true_attack == 1) & (pred == 0)).sum()
            tn_t = ((true_attack == 0) & (pred == 0)).sum()

            tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
            fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
            prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
            f1_t = 2 * prec_t * tpr_t / (prec_t + tpr_t) if (prec_t + tpr_t) > 0 else 0

            print(
                f"p{percentile:>10} {threshold:>10.4f} {tpr_t:>6.1%} "
                f"{fpr_t:>6.1%} {prec_t:>6.2f} {f1_t:>6.3f}"
            )

    def evaluate_attack_types(self) -> None:
        self.log.info("Attack type detection rates...")

        attack_labels = self.labels[
            ~self.labels.isin(["BENIGN", "Benign"]) & (self.labels.notna())
        ]

        for attack_type in sorted(attack_labels.unique()):
            mask = self.labels == attack_type
            # IF: -1 = anomaly (detected)
            detected = (self.if_predictions[mask] == -1).sum()
            total = mask.sum()
            rate = detected / total if total > 0 else 0

            status = "GOOD" if rate > 0.5 else "WARN" if rate > 0.2 else "POOR"
            print(
                f"[{status}] {attack_type[:30]:<30} {detected:>6}/{total:<6} ({rate:>6.1%})"
            )

    def save_results(self) -> None:
        self.log.info("Saving results...")

        os.makedirs("./metadata", exist_ok=True)
        os.makedirs("./artifacts", exist_ok=True)
        os.makedirs("./outputs", exist_ok=True)

        output = self.features.copy()
        output["if_score"] = self.if_scores
        output["if_prediction"] = self.if_predictions
        output["Label"] = self.labels.values

        # Filter: only attack samples detected as anomalies by IF
        attack_anomaly_mask = (output["if_prediction"] == -1) & (
            ~output["Label"].isin(["BENIGN", "Benign"])
        )

        output_filtered = output[attack_anomaly_mask]

        self.log.info(
            f"Filtered: {len(output_filtered):,} attack anomalies "
            f"(from {len(output):,} total samples)"
        )

        output_path = Path("outputs") / "deep_ae_if.csv"
        output_filtered.to_csv(output_path, index=False)
        self.log.info(f"Saved: {output_path}")

        model_ae_path = Path("artifacts") / "deep_autoencoder.pt"
        torch.save(
            {
                "model_state_dict": self.autoencoder_model.state_dict(),
                "input_dim": self.autoencoder_model.input_dim,
                "encoding_dim": self.autoencoder_model.encoding_dim,
                "layer_sizes": self.config.layer_sizes,
                "dropout_rates": self.config.dropout_rates,
                "l2_reg": self.config.l2_reg,
            },
            model_ae_path,
        )
        self.log.info(f"Saved: {model_ae_path}")

        model_if_path = Path("artifacts") / "isolation_forest.pkl"
        joblib.dump(self.isolation_forest_model, model_if_path)
        self.log.info(f"Saved: {model_if_path}")

        config_data = {
            "scaler": self.scaler,
            "clip_params": self.clip_params,
            "if_metrics": self.if_metrics,
            "encoding_dim": self.config.encoding_dim,
        }
        config_path = Path("artifacts") / "deep_ae_if_config.pkl"
        joblib.dump(config_data, config_path)

        self.log.info(f"Saved: {config_path}")

    def generate_visualizations(self) -> None:
        self.log.info("Generating visualizations...")

        os.makedirs("./plots", exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Training placeholder
        ax = axes[0, 0]
        ax.text(
            0.5,
            0.5,
            "Training completed\nwith PyTorch Lightning",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Deep AE Training History")
        ax.grid(alpha=0.3)

        # Plot 2: IF score distribution (benign vs attack)
        ax = axes[0, 1]
        bins = 50
        ax.hist(
            self.if_scores[self.test_labels == 0],
            bins=bins,
            alpha=0.7,
            label="BENIGN",
            color="green",
            density=True,
        )
        ax.hist(
            self.if_scores[self.test_labels == 1],
            bins=bins,
            alpha=0.7,
            label="Attack",
            color="red",
            density=True,
        )
        ax.set_xlabel("IF Anomaly Score")
        ax.set_title("IF Score Distribution (Latent Space)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Latent space 2D projection
        ax = axes[0, 2]
        sample_size = min(10000, len(self.latent_test))
        sample_idx = np.random.choice(len(self.latent_test), sample_size, replace=False)
        colors_scatter = [
            "red" if self.test_labels.iloc[i] == 1 else "green" for i in sample_idx
        ]
        if self.latent_test.shape[1] >= 2:
            ax.scatter(
                self.latent_test[sample_idx, 0],
                self.latent_test[sample_idx, 1],
                c=colors_scatter,
                alpha=0.3,
                s=1,
            )
            ax.set_xlabel("Latent Dim 0")
            ax.set_ylabel("Latent Dim 1")
        ax.set_title("Latent Space (First 2 Dims)")
        ax.grid(alpha=0.3)

        # Plot 4: Confusion matrix
        ax = axes[1, 0]
        cm = np.array(
            [
                [self.if_metrics["tn"], self.if_metrics["fp"]],
                [self.if_metrics["fn"], self.if_metrics["tp"]],
            ]
        )
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]:,}\n({cm[i, j] / cm.sum():.1%})"
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=10,
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticklabels(["Normal", "Attack"])
        ax.set_title("Confusion Matrix (AE->IF)")

        # Plot 5: IF score vs latent norm
        ax = axes[1, 1]
        latent_norms = np.linalg.norm(self.latent_test[sample_idx], axis=1)
        ax.scatter(
            latent_norms,
            self.if_scores[sample_idx],
            c=colors_scatter,
            alpha=0.3,
            s=1,
        )
        ax.set_xlabel("Latent Vector Norm")
        ax.set_ylabel("IF Anomaly Score")
        ax.set_title("Latent Norm vs IF Score")
        ax.grid(alpha=0.3)

        # Plot 6: Per-attack-type detection rates
        ax = axes[1, 2]
        attack_labels = self.labels[
            ~self.labels.isin(["BENIGN", "Benign"]) & (self.labels.notna())
        ]
        attack_types = sorted(attack_labels.unique())
        detection_rates = []
        attack_names = []
        for attack_type in attack_types:
            mask = self.labels == attack_type
            detected = (self.if_predictions[mask] == -1).sum()
            total = mask.sum()
            rate = detected / total if total > 0 else 0
            detection_rates.append(rate)
            attack_names.append(attack_type[:20])

        if attack_names:
            colors_bar = [
                "red" if r < 0.5 else "orange" if r < 0.8 else "green"
                for r in detection_rates
            ]
            ax.barh(range(len(attack_names)), detection_rates, color=colors_bar)
            ax.set_yticks(range(len(attack_names)))
            ax.set_yticklabels(attack_names, fontsize=8)
        ax.set_xlabel("Detection Rate")
        ax.set_title("Per-Attack Detection Rate (IF)")
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = Path("plots") / "deep_ae_if_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        self.log.info(f"Saved: {plot_path}")
        plt.close()
