import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator


class PrepareData:
    def __init__(self, config_dir: str, utility):

        self.util = utility

        # loading config parameters
        params = self.util.load_params(os.path.join(config_dir, "config.yml"))

        # creating a root directory for images
        self.train_dir = Path(params['train_dir'])
        self.test_dir = Path(params['test_dir'])
        self.phase = params['phase']
        self.validation_split = params['validation_split']
        self.input_size = params['input_size']
        self.batch_size = params['batch_size']

        self.class_names = sorted(os.listdir(self.train_dir))
        self.n_classes = len(self.class_names)

    def __image_dataframe(self):
        # creating dataframes for both train and test images
        image_dir = Path(self.train_dir) if self.phase == "train" else Path(self.test_dir)
        filepaths = list(image_dir.glob(r'**/*.jpg'))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        # creating a Pandas dataframe
        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        images = pd.concat([filepaths, labels], axis=1)

        category_samples = []
        for category in images['Label'].unique():
            category_slice = images.query("Label == @category")
            category_samples.append(category_slice)
        image_df = pd.concat(category_samples, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
        return image_df

    def create_train_val_data(self, model_type):
        # creating image data generator for train and validation datasets
        image_train_val_df = self.__image_dataframe()

        if model_type == "fine_tuned_mobile_net" or model_type == "separated_classifier":
            train_generator = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                validation_split=self.validation_split,
                rotation_range=20,
                horizontal_flip=True)

        elif model_type == "fine_tuned_EfficientNet":
            train_generator = ImageDataGenerator(
                validation_split=self.validation_split,
                rescale=1. / 255,
                rotation_range=20,
                horizontal_flip=True)

        else:
            raise Exception("Sorry, not correct model type")

            # setup dataset
        train_images = train_generator.flow_from_dataframe(
            dataframe=image_train_val_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.input_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='training')

        val_images = train_generator.flow_from_dataframe(
            dataframe=image_train_val_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.input_size,
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True,
            seed=42,
            subset='validation')

        return train_images, val_images

    def create_test_data(self):
        # creating image data generator for test dataset
        image_test_df = self.__image_dataframe()

        test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

        test_images = test_generator.flow_from_dataframe(
            dataframe=image_test_df,
            x_col='Filepath',
            y_col='Label',
            target_size=self.input_size,
            color_mode='rgb',
            classes=self.class_names,
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False)

        return test_images
