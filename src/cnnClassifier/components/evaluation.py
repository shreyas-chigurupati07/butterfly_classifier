from urllib.parse import urlparse
import pandas as pd
import os
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from pathlib import Path


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):

        valid_df = pd.read_csv(self.config.training_data_csv)

        valid_df['filename'] = valid_df['filename'].apply(
            lambda x: os.path.join(self.config.training_data_dir, x))
        


        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.3
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation='bilinear',
            shuffle=False
        )

        valid_datagenrator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)
        
        self._valid_generator = valid_datagenrator.flow_from_dataframe(
            dataframe=valid_df,
            x_col='filename',
            y_col='label',
            subset='validation',
            **dataflow_kwargs
        )
        

        
        # self._valid_generator = valid_datagenrator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset='validation',
        #     shuffle=False,
        #     **dataflow_kwargs
        # )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
        
    def evaluation(self):
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self._valid_generator)


    def save_score(self):
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path=Path('scores.json'), data=scores)