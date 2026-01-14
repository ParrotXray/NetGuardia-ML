from components.DataPreprocessing import DataPreprocessing
from components.DeepAutoencoder import DeepAutoencoder
from components.MLP import MLP
from utils.Logger import Logger
import time

if __name__ == "__main__":
    log = Logger("Main")

    log.info("Start processing data...")
    dp = DataPreprocessing()
    dp.load_datasets("./raw_data")
    dp.merge_dataset()
    dp.feature_preparation()
    dp.anomaly_detection()
    dp.output_result()

    time.sleep(3)

    log.info("Start Deep Autoencoder...")
    da = DeepAutoencoder()
    da.check_tensorflow()
    da.load_data()
    da.prepare_data()
    da.preprocess_data()
    da.build_autoencoder()
    da.train_autoencoder()
    da.calculate_ae_normalization()
    da.predict_autoencoder()
    da.train_random_forest()
    da.create_ensemble_strategies()
    da.evaluate_strategies()
    da.evaluate_attack_types()
    da.save_results()
    da.generate_visualizations()

    time.sleep(3)

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
