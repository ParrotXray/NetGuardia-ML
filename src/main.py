from method.DataPreprocessing import DataPreprocessing
from method.DeepAutoencoder import DeepAutoencoder
from model.PreprocessingConfig import PreprocessingConfig
from model.DeepAutoencoderConfig import DeepAutoencoderConfig
from utils.Logger import Logger

if __name__ == "__main__":
    log = Logger("Main")

    log.info("Start processing data...")
    dp = DataPreprocessing(PreprocessingConfig())

    # dp.load_datasets("./csv")
    # dp.merge_dataset()
    # dp.feature_preparation()
    # dp.anomaly_detection()
    # dp.output_result("./save")

    log.info("Start Deep Autoencoder...")
    da = DeepAutoencoder()
    da.check_tensorflow()

    da.load_data("../save/output_anomaly.csv")
    da.prepare_data()
    da.preprocess_data()
