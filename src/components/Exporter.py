import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import ujson
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from model import ExportConfig
from utils import Logger

from .DeepAutoencoder import AutoencoderModel
from .MLP import MLPModel


class Exporter:
    def __init__(self, config: Optional[ExportConfig] = None) -> None:
        self.deep_ae_model: Optional[AutoencoderModel] = None
        self.if_model: Optional[Any] = None
        self.mlp_model: Optional[MLPModel] = None
        self.label_encoder: Optional[Any] = None

        self.scaler: Optional[Any] = None
        self.clip_params: Optional[Dict[str, Dict[str, float]]] = None
        self.if_metrics: Optional[Dict[str, Any]] = None
        self.encoding_dim: Optional[int] = None

        self.encoder_onnx_path: Optional[Path] = None
        self.if_onnx_path: Optional[Path] = None
        self.mlp_onnx_path: Optional[Path] = None

        self.full_config: Optional[Dict[str, Any]] = None
        self.inference_config: Optional[Dict[str, Any]] = None

        self.config: ExportConfig = config or ExportConfig()
        self.log: Logger = Logger("Exporter")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_models(self) -> None:
        self.log.info("Loading models and configurations...")

        if not (
            os.path.exists("./metadata")
            or os.path.exists("./artifacts")
            or os.path.exists("./outputs")
        ):
            os.makedirs("./metadata", exist_ok=True)
            os.makedirs("./artifacts", exist_ok=True)
            os.makedirs("./outputs", exist_ok=True)

        model_path = Path("artifacts")

        try:
            ae_path = model_path / "deep_autoencoder.pt"
            checkpoint = torch.load(
                ae_path, map_location=self.device, weights_only=False
            )

            self.deep_ae_model = AutoencoderModel(
                input_dim=checkpoint["input_dim"],
                layer_sizes=checkpoint["layer_sizes"],
                encoding_dim=checkpoint["encoding_dim"],
                dropout_rates=checkpoint["dropout_rates"],
                l2_reg=checkpoint["l2_reg"],
            )
            self.deep_ae_model.load_state_dict(checkpoint["model_state_dict"])
            self.deep_ae_model.eval()
            self.log.info("Deep Autoencoder loaded (PyTorch)")
        except Exception as e:
            self.log.error(f"Failed to load Deep Autoencoder: {e}")
            raise

        try:
            if_path = model_path / "isolation_forest.pkl"
            self.if_model = joblib.load(if_path)
            self.log.info("Isolation Forest loaded")
        except Exception as e:
            self.log.error(f"Failed to load Isolation Forest: {e}")
            raise

        try:
            mlp_path = model_path / "mlp.pt"
            encoder_path = model_path / "label_encoder.pkl"

            mlp_checkpoint = torch.load(
                mlp_path, map_location=self.device, weights_only=False
            )

            self.mlp_model = MLPModel(
                input_dim=mlp_checkpoint["input_dim"],
                n_classes=mlp_checkpoint["n_classes"],
                layer_sizes=mlp_checkpoint["layer_sizes"],
                dropout_rates=mlp_checkpoint["dropout_rates"],
            )
            self.mlp_model.load_state_dict(mlp_checkpoint["model_state_dict"])
            self.mlp_model.eval()

            self.label_encoder = joblib.load(encoder_path)
            self.log.info("MLP Improved loaded (PyTorch)")
        except Exception as e:
            self.log.error(f"Failed to load MLP: {e}")
            raise

        try:
            config_path = model_path / "deep_ae_if_config.pkl"
            ae_if_config = joblib.load(config_path)
            self.scaler = ae_if_config["scaler"]
            self.clip_params = ae_if_config["clip_params"]
            self.if_metrics = ae_if_config.get("if_metrics", None)
            self.encoding_dim = ae_if_config.get("encoding_dim", 16)
            self.log.info("AE-IF configuration loaded")

            if self.if_metrics:
                print("IF detection metrics:")
                self.log.info(f"TPR: {self.if_metrics['tpr']:.2%}")
                self.log.info(f"FPR: {self.if_metrics['fpr']:.2%}")
                self.log.info(f"Precision: {self.if_metrics['precision']:.3f}")
                self.log.info(f"F1: {self.if_metrics['f1']:.3f}")
        except Exception as e:
            self.log.error(f"Failed to load configuration: {e}")
            raise

    def export_encoder_onnx(self) -> None:
        self.log.info("Converting AE Encoder to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        self.deep_ae_model.eval()
        input_dim = self.deep_ae_model.input_dim
        dummy_input = torch.randn(1, input_dim)

        # Export only the encoder portion
        encoder_wrapper = self.deep_ae_model.encoder

        self.encoder_onnx_path = Path("exports") / "encoder.onnx"

        torch.onnx.export(
            encoder_wrapper,
            (dummy_input,),
            self.encoder_onnx_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["latent"],
            dynamo=False,
            external_data=False,
        )

        self.log.info(f"Saved: {self.encoder_onnx_path}")

        onnx_model = onnx.load(self.encoder_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

    def export_if_onnx(self) -> None:
        self.log.info("Converting Isolation Forest to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        initial_type = [("float_input", FloatTensorType([None, self.encoding_dim]))]

        onx = convert_sklearn(
            self.if_model,
            initial_types=initial_type,
            target_opset=self.config.opset_version,
        )

        self.if_onnx_path = Path("exports") / "isolation_forest.onnx"
        with open(self.if_onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        self.log.info(f"Saved: {self.if_onnx_path}")

        onnx_model = onnx.load(self.if_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

    def export_mlp_onnx(self) -> None:
        self.log.info("Converting MLP Classifier to ONNX...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        self.mlp_model.eval()
        input_dim = self.mlp_model.input_dim
        dummy_input = torch.randn(1, input_dim)

        mlp_with_softmax = nn.Sequential(self.mlp_model, nn.Softmax(dim=1))

        self.mlp_onnx_path = Path("exports") / "mlp.onnx"

        torch.onnx.export(
            mlp_with_softmax,
            (dummy_input,),
            self.mlp_onnx_path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            external_data=False,
        )

        self.log.info(f"Saved: {self.mlp_onnx_path}")

        onnx_model = onnx.load(self.mlp_onnx_path)
        onnx.checker.check_model(onnx_model)
        self.log.info("ONNX model validation passed")

        print("\nMLP ONNX outputs:")
        for output in onnx_model.graph.output:
            print(f"  {output.name}: {output.type}")

    def build_config_json(self) -> None:
        self.log.info("Building configuration JSON...")

        if not os.path.exists("./exports"):
            os.makedirs("./exports", exist_ok=True)

        scaler_params = {
            "mean": self.scaler.mean_.tolist(),
            "std": self.scaler.scale_.tolist(),
            "feature_names": (
                self.scaler.feature_names_in_.tolist()
                if hasattr(self.scaler, "feature_names_in_")
                else []
            ),
        }

        clip_params_json = {}
        for col, params in self.clip_params.items():
            clip_params_json[col] = {
                "lower": float(params["lower"]),
                "upper": float(params["upper"]),
            }

        if_metrics_json = {}
        if self.if_metrics:
            if_metrics_json = {
                "tpr": float(self.if_metrics["tpr"]),
                "fpr": float(self.if_metrics["fpr"]),
                "precision": float(self.if_metrics["precision"]),
                "f1": float(self.if_metrics["f1"]),
            }

        attack_labels = {
            str(i): label for i, label in enumerate(self.label_encoder.classes_)
        }

        self.full_config = {
            "created_at": pd.Timestamp.now().isoformat(),
            "framework": "PyTorch",
            "architecture": "AE_Encoder -> IsolationForest (latent space)",
            "model": {
                "encoder": {
                    "file": "encoder.onnx",
                    "input_dim": int(self.deep_ae_model.input_dim),
                    "encoding_dim": int(self.encoding_dim),
                },
                "isolation_forest": {
                    "file": "isolation_forest.onnx",
                    "n_estimators": int(self.if_model.n_estimators),
                    "contamination": float(self.if_model.contamination),
                    "latent_dim": int(self.encoding_dim),
                },
                "mlp_classifier": {
                    "file": "mlp.onnx",
                    "input_dim": int(self.mlp_model.input_dim),
                    "n_classes": int(len(self.label_encoder.classes_)),
                },
            },
            "preprocessing": {
                "clip_params": clip_params_json,
                "scaler": scaler_params,
                "post_scaling_clip": {
                    "min": self.config.post_scaling_clip_min,
                    "max": self.config.post_scaling_clip_max,
                },
            },
            "if_metrics": if_metrics_json,
            "attack_labels": attack_labels,
            "feature_order": scaler_params["feature_names"],
        }

        self.inference_config = {
            "architecture": "AE_Encoder -> IsolationForest (latent space)",
            "clip_params": clip_params_json,
            "scaler_mean": scaler_params["mean"],
            "scaler_std": scaler_params["std"],
            "post_clip_min": self.config.post_scaling_clip_min,
            "post_clip_max": self.config.post_scaling_clip_max,
            "encoding_dim": int(self.encoding_dim),
            "if_metrics": if_metrics_json,
            "attack_labels": attack_labels,
            "feature_names": scaler_params["feature_names"],
        }

    def save_config_json(self) -> None:
        self.log.info("Saving configuration JSON...")

        os.makedirs("./exports", exist_ok=True)

        full_config_path = Path("exports") / "full_config.json"
        with open(full_config_path, "w", encoding="utf-8") as f:
            ujson.dump(self.full_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {full_config_path}")

        inference_config_path = Path("exports") / "inference_config.json"
        with open(inference_config_path, "w", encoding="utf-8") as f:
            ujson.dump(self.inference_config, f, indent=2, ensure_ascii=False)
        self.log.info(f"Saved: {inference_config_path} (inference only)")

    def verify_onnx_models(self) -> None:
        self.log.info("Verifying ONNX models...")

        n_features = len(self.clip_params)
        test_input = np.random.randn(1, n_features).astype(np.float32)

        scaler_params = self.full_config["preprocessing"]["scaler"]
        clip_params_json = self.full_config["preprocessing"]["clip_params"]

        for i, col in enumerate(scaler_params["feature_names"]):
            if col in clip_params_json:
                test_input[0, i] = np.clip(
                    test_input[0, i],
                    clip_params_json[col]["lower"],
                    clip_params_json[col]["upper"],
                )

        test_input_scaled = (test_input - np.array(scaler_params["mean"])) / np.array(
            scaler_params["std"]
        )
        test_input_scaled = np.clip(
            test_input_scaled,
            self.config.post_scaling_clip_min,
            self.config.post_scaling_clip_max,
        ).astype(np.float32)

        # Step 1: Encoder (feature extraction -> latent space)
        print("Testing AE Encoder:")
        session_encoder = ort.InferenceSession(str(self.encoder_onnx_path))
        latent_output = session_encoder.run(None, {"input": test_input_scaled})[0]
        print(f"Input dim: {test_input_scaled.shape[1]}")
        print(f"Latent dim: {latent_output.shape[1]}")
        print(f"Latent values: {latent_output[0][:5]}...")

        # Step 2: Isolation Forest (anomaly detection in latent space)
        print()
        print("Testing Isolation Forest (latent space):")
        session_if = ort.InferenceSession(str(self.if_onnx_path))
        if_output = session_if.run(None, {"float_input": latent_output})
        if_label = if_output[0][0]
        if_scores = if_output[1]

        print(
            f"IF Prediction: {if_label} ({'anomaly' if if_label == -1 else 'normal'})"
        )
        print(f"IF Score output: {if_scores[0]}")

        is_anomaly = if_label == -1

        # Step 3: MLP (attack classification if anomaly)
        print()
        print("Testing MLP Classifier:")
        session_mlp = ort.InferenceSession(str(self.mlp_onnx_path))
        mlp_output = session_mlp.run(None, {"input": test_input_scaled})[0]

        mlp_probs = np.exp(mlp_output[0]) / np.sum(np.exp(mlp_output[0]))
        predicted_class = np.argmax(mlp_probs)
        confidence = mlp_probs[predicted_class]

        print(
            f"Predicted Class: {predicted_class} ({self.label_encoder.classes_[predicted_class]})"
        )
        print(f"Confidence: {confidence:.6f}")

        # Pipeline summary
        print()
        print("Pipeline Result:")
        print(
            f"  Input ({n_features}D) -> Encoder -> Latent ({latent_output.shape[1]}D) -> IF -> {'ANOMALY' if is_anomaly else 'NORMAL'}"
        )
        if is_anomaly:
            print(
                f"  Attack Type: {self.label_encoder.classes_[predicted_class]} (Confidence: {confidence:.2%})"
            )
        else:
            print(f"  Predicted as Normal Traffic")
        print()

    def print_summary(self) -> None:
        self.log.info("Export Summary...")

        print("Model Information:")
        input_dim = self.deep_ae_model.input_dim
        print(f"Framework: PyTorch + PyTorch Lightning")
        print(f"Architecture: AE Encoder -> IF (Latent Space)")
        print(f"Encoder: {input_dim}D -> {self.encoding_dim}D latent space")
        print(
            f"Isolation Forest: {self.if_model.n_estimators} trees on {self.encoding_dim}D latent"
        )
        print(f"MLP: {len(self.label_encoder.classes_)} attack classes")

        if self.if_metrics:
            print()
            print("Detection Metrics (AE->IF):")
            print(f"TPR: {self.if_metrics['tpr']:.2%}")
            print(f"FPR: {self.if_metrics['fpr']:.2%}")
            print(f"Precision: {self.if_metrics['precision']:.3f}")
            print(f"F1-Score: {self.if_metrics['f1']:.3f}")

        print()
        print("Inference Pipeline:")
        print(f"  1. Preprocess: clip + scale ({input_dim} features)")
        print(f"  2. Encode: AE encoder ({input_dim}D -> {self.encoding_dim}D)")
        print(f"  3. Detect: IF on latent space (normal/anomaly)")
        print(f"  4. Classify: MLP on original features (attack type)")
