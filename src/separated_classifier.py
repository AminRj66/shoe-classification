import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class SeparatedClassifier:
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
        self.variance_ratio = params['variance_ratio']
        self.CV_fold = params['CV_fold']
        self.svm_C = params['svm']['C']
        self.svm_gamma = params['svm']['gamma']

    @staticmethod
    def __load_pretrained_model(input_shape):
        # defining a CNN using the MobileNetV2 architecture pre-trained on the ImageNet dataset
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=list(input_shape),
            include_top=False,
            # weights='imagenet',
            weights="src/weights/mobilenet_v2_weights.h5",
            pooling='avg')

        pretrained_model.summary()

        return pretrained_model

    def __create_model(self):
        # defining architecture for training phase and also loading pretrained model for testing phase
        pretrained_model = self.__load_pretrained_model(self.input_shape)
        last_conv_layer_name = 'block_16_project_BN'  # layer name based on the model's summary
        # Create a feature extraction model
        feature_extractor_model = tf.keras.Model(inputs=pretrained_model.input,
                                                 outputs=pretrained_model.get_layer(last_conv_layer_name).output)
        # Freeze the feature extractor model
        feature_extractor_model.trainable = False
        if self.training:
            # PCA with variance-based selection of principal components
            pca = PCA(n_components=self.variance_ratio)

            # SVM with a nonlinear RBF kernel
            svm_classifier = SVC(kernel="rbf")  # Nonlinear RBF kernel

            # Create a pipeline with PCA and SVM
            pipeline = Pipeline([
                ('pca', pca),
                ('svm', svm_classifier)
            ])

        else:
            pipeline = self.util.load_pickle(self.save_path, self.model_type)

        return feature_extractor_model, pipeline

    def train_model(self, train_images: pd.DataFrame):
        feature_extractor, pipeline = self.__create_model()
        img_features = feature_extractor.predict(train_images)

        # Define the hyperparameters grid to search over
        param_grid = {
            'svm__C': self.svm_C,  # Regularization parameter
            'svm__gamma': self.svm_gamma  # Kernel coefficient for RBF
        }

        # Perform Grid Search with cross-validation to find the best hyperparameters
        grid_search = GridSearchCV(pipeline, param_grid, cv=self.CV_fold)  # cross-validation
        img_features = img_features.reshape(len(train_images.labels), -1)
        print("Grid Search optimization on cross validated data set has started ...")
        grid_search.fit(img_features, train_images.labels)

        # Retrieve the best model after hyperparameter tuning
        print("Selecting best estimator has started ...")
        best_model = grid_search.best_estimator_

        # Calculate accuracy based on predictions
        predictions = best_model.predict(img_features)  # Predict on the training data
        accuracy = accuracy_score(train_images.labels, predictions)
        print(f"Training Accuracy: {accuracy}")

        # Get the learning curve
        train_sizes, train_scores, valid_scores = learning_curve(best_model, img_features, train_images.labels,
                                                                 train_sizes=np.linspace(self.svm_C[0],
                                                                                         self.svm_C[1],
                                                                                         self.svm_C[2]),
                                                                 cv=self.CV_fold)
        history = {"train_sizes": train_sizes, "train_scores": train_scores, "valid_scores": valid_scores}

        # Save the best model to a file for future testing
        self.util.save_pickle(best_model, self.save_path, self.model_type)

        return history, accuracy

    def test_model(self, test_images: pd.DataFrame):
        # testing a pre-saved model
        feature_extractor, pipeline = self.__create_model()
        img_features = feature_extractor.predict(test_images)
        img_features = img_features.reshape(len(test_images.labels), -1)

        predictions = pipeline.predict(img_features)

        return predictions

    @staticmethod
    def plot_training_performance(history):
        # Calculate the mean and standard deviation of the training and validation scores
        train_scores_mean = np.mean(history['train_scores'], axis=1)
        train_scores_std = np.std(history['train_scores'], axis=1)
        valid_scores_mean = np.mean(history['valid_scores'], axis=1)
        valid_scores_std = np.std(history['valid_scores'], axis=1)

        # Plot the training curve
        plt.figure()
        plt.xlabel("Training Samples")
        plt.ylabel("Score")
        plt.grid()
        plt.fill_between(history['train_sizes'], train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="r")
        plt.fill_between(history['train_sizes'], valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std,
                         alpha=0.1, color="g")
        plt.plot(history['train_sizes'], train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(history['train_sizes'], valid_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.show()

    def plot_confusion_metrics(self, test_images, class_names):
        n_classes = len(class_names)
        # Make predictions on the new test dataset
        _, predictions_test = self.test_model(test_images)

        # Calculate and plot the confusion matrix
        cm = confusion_matrix(test_images.labels, predictions_test)  # Compute confusion matrix
        plt.figure(figsize=(30, 30))
        sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
        plt.xticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=90)
        plt.yticks(ticks=np.arange(n_classes) + 0.5, labels=test_images.class_indices, rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        print(classification_report(test_images.labels, predictions_test, target_names=class_names, digits=4))

        plt.show()

    def run_separated_classifier(self, data_object):
        if self.phase == "train":
            train_images, val_images = data_object.create_train_val_data(self.model_type)
            history, accuracy = self.train_model(train_images)
            self.plot_training_performance(history)

            test_images = data_object.create_test_data()
            self.plot_confusion_metrics(test_images, data_object.class_names)

        else:
            test_images = data_object.create_test_data()
            self.plot_confusion_metrics(test_images, data_object.class_names)
