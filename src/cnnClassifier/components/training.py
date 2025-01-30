from cnnClassifier.entity.config_entity import TrainingConfig
from pathlib import Path
import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time
import pandas as pd
import numpy as np


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        # ✅ Force-enable eager execution at the beginning
        tf.config.run_functions_eagerly(True)
        print(f"✅ Eager Execution Enabled: {tf.executing_eagerly()}\n")


    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
        # ✅ Debugging Step: Print Model Output
        print(f"✅ Model Loaded: Output shape {self.model.output_shape}")

        # ✅ Ensure output layer matches number of classes
        expected_classes = self.config.params_classes

        if self.model.output_shape[-1] != expected_classes:
            print(f"🔄 Rebuilding Model: Expected {expected_classes} classes, found {self.model.output_shape[-1]}")
            
            # ✅ Remove the last layer
            base_model = tf.keras.models.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
            
            # ✅ Add new classification head
            new_output = tf.keras.layers.Dense(expected_classes, activation="softmax")(base_model.output)
            self.model = tf.keras.models.Model(inputs=base_model.input, outputs=new_output)

            print(f"✅ Model rebuilt with {expected_classes} output classes")

        # ✅ Compile the Model Again
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("✅ Model Recompiled Successfully")



    # def get_base_model(self):
    #     self.model = tf.keras.models.load_model(
    #         self.config.updated_base_model_path
    #     )


    def train_valid_generator(self):

        # ✅ Debugging: Print expected file path
        abs_path = os.path.abspath(self.config.training_data_csv)
        print(f"🔍 Checking file existence: {abs_path}")

        if not os.path.exists(self.config.training_data_csv):
            raise FileNotFoundError(f"❌ CSV file not found at {self.config.training_data_csv}\n"
                                    f"🔹 Expected Absolute Path: {abs_path}")

        print(f"✅ CSV file found: {abs_path}")
        train_df = pd.read_csv(self.config.training_data_csv)

        train_df['filename'] = train_df['filename'].apply( 
            lambda x: os.path.join(self.config.training_data_dir, x)
        )

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.2
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode='categorical',
            interpolation='bilinear'
        ) 

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs)
        
        self.valid_generator = valid_datagenerator.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='label',
            subset='validation',
            shuffle=False,
            **dataflow_kwargs
        )

        # self.valid_generator = valid_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset='validation',
        #     shuffle=False,
        #     **dataflow_kwargs
        # )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='label',
            subset='training',
            shuffle=True,
            **dataflow_kwargs
        )
        print(f'Found {self.train_generator.samples} training images')
        print(f'Found {self.valid_generator.samples} validation images')
        
        # self.train_generator = train_datagenerator.flow_from_directory(
        #     directory=self.config.training_data,
        #     subset='training',
        #     shuffle=True,
        #     **dataflow_kwargs
        # )

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)

    def train(self, callback_list: list):
        # ✅ Ensure optimizer is recompiled
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        print(f"🚀 Training Started for {self.config.params_epochs} epochs with {self.steps_per_epoch} steps per epoch")

        # ✅ Train the model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        print("✅ Training Complete!")
        self.save_model(path=self.config.trained_model_path, model=self.model)



    # def train(self,callback_list: list):
    #     self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
    #     self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

    #     self.model.fit(
    #         self.train_generator,
    #         epochs=self.config.params_epochs,
    #         steps_per_epoch=self.steps_per_epoch,
    #         validation_steps=self.validation_steps,
    #         validation_data=self.valid_generator,
    #         callbacks=callback_list
    #     )

    #     self.save_model(path=self.config.training_model_path, model=self.model)
