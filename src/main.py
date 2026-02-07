#!/usr/bin/env python3

import argparse
import time
from datetime import timedelta

from components import MLP, DataPreprocess, DeepAutoencoder, Exporter
from utils import Logger

if __name__ == "__main__":
    log = Logger("Main")

    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="Training net-guardia models")

    parser.add_argument("-s", "--set", help="Select Dataset year")

    parser.add_argument("-a", "--all", action="store_true", help="ALL")
    parser.add_argument(
        "-dp", "--datapreprocess", action="store_true", help="DataPreprocess"
    )
    parser.add_argument(
        "-da", "--deepautoencoder", action="store_true", help="DeepAutoencoder"
    )
    parser.add_argument("-mp", "--mlp", action="store_true", help="MLP")
    parser.add_argument("-ep", "--export", action="store_true", help="Export")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()

    if args.all or args.datapreprocess:
        log.info("Start processing data...")
        dp = DataPreprocess(args.set)
        dp.load_datasets()
        dp.statistics_dataset()
        dp.feature_preparation()
        dp.output_result()

        time.sleep(3)

    if args.all or args.deepautoencoder:
        log.info("Start Deep Autoencoder + Isolation Forest...")
        da = DeepAutoencoder()
        da.check_tensorflow()
        da.load_data()
        da.prepare_data()
        da.preprocess_data()
        da.build_autoencoder()
        da.train_autoencoder()
        da.extract_latent_features()
        da.train_isolation_forest()
        da.predict_isolation_forest()
        da.evaluate_detection()
        da.evaluate_attack_types()
        da.save_results()
        da.generate_visualizations()

        time.sleep(3)

    if args.all or args.mlp:
        log.info("Start MLP...")
        mlp = MLP()
        mlp.load_data()
        mlp.prepare_features()
        mlp.split_data()
        mlp.apply_smote()
        mlp.calculate_class_weights()
        mlp.build_model()
        mlp.train_model()
        mlp.evaluate_model()
        mlp.save_results()
        mlp.generate_visualizations()

        time.sleep(3)

    if args.all or args.export:
        log.info("Export models to onnx...")
        ep = Exporter()
        ep.load_models()
        ep.export_encoder_onnx()
        ep.export_if_onnx()
        ep.export_mlp_onnx()
        ep.build_config_json()
        ep.save_config_json()
        ep.verify_onnx_models()
        ep.print_summary()

    end = time.perf_counter()

    log.info(f"Execution time: {timedelta(seconds=(end - start))}")
