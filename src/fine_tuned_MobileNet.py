import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class FineTunedMobileNet:
    def __init__(self, config_dir: str, n_classes: int, utility):

        self.model = None
        self.util = utility

        # loading config parameters
        params = self.util.load_params(os.path.join(config_dir, "config.yml"))

        self.n_classes = n_classes
        self.phase = params['phase']
        self.model_type = params['model_type']
        self.save_path = params['saved_models_dir']
        self.training = params['phase'] == "train"
        self.model_name = params['model_name']
        self.input_shape = tuple(params['input_size'] + [3])
        self.batch_size = params['batch_size']
        self.lr = params['learning_rate']
        self.epochs = params['epochs']

    @staticmethod
    def __load_pretrained_model(input_shape):
        # defining a CNN using the MobileNetV2 architecture pre-trained on the ImageNet dataset
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=list(input_shape),
            include_top=False,
            # weights='imagenet',
            weights="src/weights/mobilenet_v2_weights.h5",
            pooling='avg')

        return pretrained_model

    def __create_model(self):
        # defining architecture for training phase and also loading pretrained model for testing phase
        if self.training:
            pretrained_model = self.__load_pretrained_model(self.input_shape)
            pretrained_model.trainable = False

            # customizing the inference part of the model
            model = tf.keras.models.Sequential([
                # layers.Input(shape=self.input_shape),
                # layers.RandomRotation(0.125),
                pretrained_model,
                # tf.keras.layers.GlobalAveragePooling2D(),
                # layers.Dropout(0.25),
                # layers.Dense(256, activation='relu'),
                # layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                layers.Dense(self.n_classes, activation='softmax')])

            # configuring the learning process using Adam optimizer
            model.compile(optimizer=Adam(learning_rate=self.lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['Accuracy'])
            # model.build()
            print(model.summary())

        else:
            model = self.util.load_model(self.save_path, self.model_name)
            model.compile(optimizer=Adam(learning_rate=self.lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=['Accuracy'])

        return model

    def train_model(self, train_images: pd.DataFrame, val_images: pd.DataFrame):
        self.model = self.__create_model()
        # training the model with a predefined batch size
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint("best_model_" + self.model_type + ".h5", monitor='val_loss',
                                           save_best_only=True, mode='min')
        history_data = self.model.fit(train_images,
                                      steps_per_epoch=train_images.samples // self.batch_size // 2,
                                      epochs=self.epochs,
                                      validation_data=val_images,
                                      validation_steps=val_images.samples // self.batch_size // 2,
                                      verbose=1,
                                      callbacks=[early_stop, model_checkpoint],
                                      shuffle=True)

        self.util.save_model(self.model, self.save_path, self.model_type)

        return history_data.history

    def test_model(self, test_images: pd.DataFrame):
        # testing a pre-saved model
        self.model = self.__create_model()
        loss, auc = self.model.evaluate(test_images)

        predictions = self.model.predict(test_images)  # Predict on the training data
        accuracy = accuracy_score(test_images.labels, predictions)
        print(f"Training Accuracy: {accuracy}")

        return loss, auc

    @staticmethod
    def plot_training_performance(epochs, data, train_param, val_param):
        plt.figure(figsize=(10, 7))

        plt.plot(epochs, data[train_param], 'g', label=f'Training ({train_param})')
        plt.plot(epochs, data[val_param], 'red', label=f'Validation ({val_param})')

        plt.title("Training performance")
        plt.xlabel('Epochs')
        plt.ylabel(train_param)

        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_metrics(model, test_images, class_names):
        n_classes = len(class_names)
        predictions = np.argmax(model.predict(test_images), axis=1)
        cm = confusion_matrix(test_images.labels, predictions)
        plt.figure(figsize=(30, 30))
        sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
        plt.xticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=90)
        plt.yticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        print(classification_report(test_images.labels, predictions, target_names=class_names, digits=4))

        plt.show()

    def run_fine_tuned_mobile_net(self, data_object):
        if self.phase == "train":
            train_images, val_images = data_object.create_train_val_data(self.model_type)
            history = self.train_model(train_images, val_images)

            epochs = range(1, len(history['Accuracy']) + 1)
            self.plot_training_performance(epochs, history, 'Accuracy', 'val_Accuracy')
            self.plot_training_performance(epochs, history, 'loss', 'val_loss')

            test_images = data_object.create_test_data()
            self.plot_confusion_metrics(self.model, test_images, data_object.class_names)

        else:
            test_images = data_object.create_test_data()

            self.plot_confusion_metrics(self.model, test_images, data_object.class_names)
