"""Train model script

Script to train model over a input filepath

"""

import logging
from pathlib import Path, PurePath

import pandas as pd

from src.utils.modelling import ArtifactSaverLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]


class BatchInference(object):
    def __init__(self, input_filepath, output_filepath, model_filepath, model_name):
        self.input_filepath = self._load_path(input_filepath)
        self.output_filepath = self._load_path(output_filepath)
        self.model_filepath = self._load_path(model_filepath)
        self.model_name = model_name

    @staticmethod
    def _load_path(path):
        if isinstance(path, str):
            pathlib_path = Path(path)
        elif isinstance(path, PurePath):
            pathlib_path = path
        else:
            raise ValueError("Not able to create Path from str")
        return pathlib_path

    def load_data(self):
        if self.input_filepath.is_file():
            data = pd.read_csv(self.input_filepath)
        else:
            raise FileNotFoundError("Modelling file note found")
        return data

    def load_model(self):
        if self.model_filepath.joinpath(self.model_name).is_file():
            artifact_interface = ArtifactSaverLoader(
                models_filepath=self.model_filepath
            )
            model = artifact_interface.load_artifact(artifact_name=self.model_name)
        else:
            raise FileNotFoundError("Artifact does not exist")
        return model

    def write_to_predictions(self, df):
        df.to_csv(self.output_filepath, index=False)
        return self.output_filepath.is_file()

    def predict(self):
        logger = logging.getLogger(__name__)

        logger.info(f"Loading data from {self.input_filepath}")
        data_df = self.load_data()

        logger.info(f"Loading model from {self.model_filepath} - {self.model_name}")
        model = self.load_model()

        logger.info(f"Run predictions")
        data_df["predicted_price"] = model.predict(data_df)

        self.write_to_predictions(data_df)

        logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = PROJECT_DIR.joinpath("data/processed/test_data.csv")
    output_filepath = PROJECT_DIR.joinpath("results/predictions.csv")

    model_filepath = PROJECT_DIR.joinpath("models")
    model_name = "linear-sdg-reg-2020-11-22T17H-22M.p"

    predictor = BatchInference(
        input_filepath, output_filepath, model_filepath, model_name
    )

    predictor.predict()
