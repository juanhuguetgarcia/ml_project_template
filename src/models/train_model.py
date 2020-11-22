"""Train model script

Script to train model over a input filepath

"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.modelling import ArtifactSaverLoader


PROJECT_DIR = Path(__file__).resolve().parents[2]


class ModelTrainer(object):
    def __init__(self, input_filepath, output_filepath, target_name, model_name, test_ratio=0.2,):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.test_ratio = test_ratio
        self.target_name = target_name
        self.model_name = model_name

    def load_modelling_data(self):
        if isinstance(self.input_filepath, str):
            self.input_filepath = Path(self.input_filepath)
        if self.input_filepath.is_file():
            data = pd.read_csv(self.input_filepath)
        else:
            raise FileNotFoundError('Modelling file note found')
        return data

    def get_x_y(self, df):
        df = df.sample(frac=1)
        X = df.drop(columns=self.target_name)
        y = df[self.target_name]
        return X, y

    def split_data(self, data_df):
        X, y = self.get_x_y(data_df)
        return train_test_split(X, y, test_size=self.test_ratio)

    @staticmethod
    def fit_pipeline(X, y):
        scaling_step = ('scaler', StandardScaler())
        regression_step = ('regressor', LinearRegression())
        steps = [scaling_step, regression_step]
        pipe = Pipeline(steps)
        pipe.fit(X, y)
        return pipe

    @staticmethod
    def evaluate_pipeline(pipe, X_train, X_test, y_train, y_test):
        y_pred_in_sample = pipe.predict(X_train)
        y_pred_out_of_sample = pipe.predict(X_test)
        in_sample_error = np.sqrt(mean_squared_error(y_train, y_pred_in_sample))
        out_of_sample_error = np.sqrt(mean_squared_error(y_test, y_pred_out_of_sample))
        evaluation = {'in_sample_error': in_sample_error, 'out_of_sample_error': out_of_sample_error}
        return evaluation

    def train(self):
        logger = logging.getLogger(__name__)

        logger.info(f'Loading data from {self.input_filepath}')
        data_df = self.load_modelling_data()

        logger.info(f'Train - Test split with test ratio = {self.test_ratio}')
        X_train, X_test, y_train, y_test = self.split_data(data_df)

        logger.info(f'Fit pipeline')
        trained_pipeline = self.fit_pipeline(X_train, y_train)

        logger.info('Evaluate pipeline')
        evaluation = self.evaluate_pipeline(trained_pipeline, X_train, X_test, y_train, y_test)
        logger.info(f"Mean error in-sample: {evaluation['in_sample_error']}")
        logger.info(f"Mean error out-sample: {evaluation['out_of_sample_error']}")

        logger.info(f'Retraining with all data')
        X, y = self.get_x_y(data_df)
        final_pipeline = self.fit_pipeline(X, y)

        logger.info('Save artifacts')
        artifact_interface = ArtifactSaverLoader(models_filepath=self.output_filepath)
        artifact_interface.save_artifact(artifact=final_pipeline, artifact_name=self.model_name)

        logger.info('Done!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = PROJECT_DIR.joinpath('data/processed/modelling_data.csv')
    output_filepath = PROJECT_DIR.joinpath('models')
    model_name = 'linear-sdg-reg'
    target_name = 'price'

    trainer = ModelTrainer(input_filepath, output_filepath, target_name, model_name, test_ratio=0.2)

    trainer.train()
