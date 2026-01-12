from DataPreprocessing import PreprocessingConfig, DataPreprocessing
from DeepAutoencoder import DeepAutoencoder
from Logger import Logger

if __name__ == "__main__":
    log = Logger("Main")

    log.info("Start processing data...")
    dp = DataPreprocessing(PreprocessingConfig())

    dp.load_datasets("./csv")
    dp.merge_dataset()
    dp.feature_preparation()
    dp.anomaly_detection()
    dp.output_result("./save")

    log.info("Start Deep Autoencoder...")
    da = DeepAutoencoder()
    da.load_dataset("./save/output_anomaly.csv")
