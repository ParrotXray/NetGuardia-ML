from components.DataPreprocessing import DataPreprocessing
from components.DeepAutoencoder import DeepAutoencoder
from components.MLP import MLP
from utils.Logger import Logger

if __name__ == "__main__":
    log = Logger("Main")

    log.info("Start processing data...")
    dp = DataPreprocessing()

    # dp.load_datasets("./csv")
    # dp.merge_dataset()
    # dp.feature_preparation()
    # dp.anomaly_detection()
    # dp.output_result("./save")

    log.info("Start Deep Autoencoder...")
    da = DeepAutoencoder()
    # da.check_tensorflow()
    # da.load_data("../save/output_anomaly.csv")
    # da.prepare_data()
    # da.preprocess_data()
    # da.build_autoencoder()
    # da.train_autoencoder()
    # da.calculate_ae_normalization()
    # da.predict_autoencoder()
    # da.train_random_forest()
    # da.create_ensemble_strategies()
    # da.evaluate_strategies()
    # da.evaluate_attack_types()
    # da.save_results("../save")
    # da.generate_visualizations("../save")

    log.info("Start MLP...")
    mlp = MLP()
    mlp.load_data("../save/output_deep_ae_ensemble.csv", "../save/deep_ae_ensemble_config.pkl")
    mlp.prepare_features()
    mlp.split_data()
    mlp.apply_smote()
    mlp.calculate_class_weights()
    mlp.build_model()
    mlp.train_model()
    mlp.evaluate_model()
    mlp.save_results("../save")
    mlp.generate_visualizations("../save")
