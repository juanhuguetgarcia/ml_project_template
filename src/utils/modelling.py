"""Modelling

Utils to use during modelling stage
"""

import pickle
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]


class ArtifactSaverLoader(object):

    def __init__(self, models_filepath=PROJECT_DIR.joinpath('models')):
        self.models_filepath = models_filepath

    def save_artifact(self, artifact, artifact_name=None):
        """Persists artifact as a pickle
        """
        today = datetime.today()
        today_str = datetime.strftime(today, '%Y-%m-%dT%H-%M-%SZ')
        artifact_unique_name = f'{artifact_name}-{today_str}.p' if artifact_name else f'artifact-{today_str}.p'

        artifact_filepath = self.models_filepath.joinpath(artifact_unique_name)
        pickle.dump(artifact, open(artifact_filepath, 'wb'))
        return artifact_filepath.is_file()

    def load_artifact(self, artifact_name):
        artifact_filepath = self.models_filepath.joinpath(artifact_name)
        if artifact_filepath.is_file():
            model = pickle.load(open(artifact_filepath, 'rb'))
        else:
            raise FileNotFoundError('Artifact does not exist')
        return model
