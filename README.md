# ***Shoes Classification Task***


# 1. Description

The AI system being developed is a shoe classification model trained on a dataset comprising six classes of shoes: boots, sneakers, flip-flops, loafers, sandals, and soccer shoes. Each shoe category is represented by 249 images, with varying sizes but all images are in JPEG format. The dataset encapsulates a broad range of shoe variations, capturing the intricacies of each shoe type. From different perspectives and environments to various styles and designs, the dataset offers a comprehensive representation of shoes, enriching the training data for the classification model. This diversity ensures that the AI system is exposed to a wide array of shoe attributes, enhancing its ability to generalize and accurately classify shoes across different categories.


# 2. Prerequisites

Before running the code, make sure you have the following installed: <br>

tensorflow==2.12.1 <br>
keras==2.12.0 <br>
matplotlib==3.7.2 <br>
numpy==1.24.3 <br>
pandas==2.0.3 <br>
PyYAML==6.0.1 <br>
scikit_learn==1.3.0 <br>
seaborn==0.12.2 <br>
joblib~=1.4.2 <br>
scikit-learn~=1.3.0 <br>


# 3. Getting Started

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set up the configuration parameters in the `config/config.yml` file according to your needs.

# 4. Running Via Docker Image

1. Using the Dockerfile in the project directory with the necessary instructions, build the image file via following command: <br>
`docker build -t shoes_classification`.
2. Once the image is built successfully, you can run a container from the image using the following command: <br>
`docker run shoes_classification`.


# 5. Image Classification

This repository is a collection of code that implements a deep learning model capable of performing multi-class image classification. The first architecture, i.e. fine-tuned MobileNet_v2 showcases a neural network model constructed using TensorFlow's Keras API, featuring a pretrained MobileNetV2 as the base network. In the customization phase, a sequential model is constructed to extend the base MobileNetV2 architecture. This extension includes a dropout layer with a 0.1 dropout rate to mitigate overfitting by randomly deactivating a fraction of neurons during training. Subsequently, a dense layer with 64 units and a ReLU activation function has been added to introduce non-linearity and enhance the model's capacity to capture complex patterns within the data. 
The code allows for the training and testing of the model on a given dataset of shoe images. During training, the model utilizes a two-step process. First, the feature extraction section of the model, which is pretrained, is utilized. This section has been previously trained on a large dataset and is not updated during the training process. This allows the model to benefit from the knowledge learned from a diverse range of images.
The second step of the training process involves re-training the semantic encoding section of the model. This section consists of a fully connected network at the end of the model. During this step, the model is trained specifically on current images. This allows the model to learn the specific features and characteristics of shoe images, enhancing its ability to accurately classify them.

The implementation covers three CNN based models, including: <br>
1. **End-to-end finetuning MobileNet_v2 classifier:** <br>
This model achieves the highest performance among all models, and the results will be plotted based on this architecture from now on.
 <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model1_architecture.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model1_architecture.png">
  </a>
</p>

 <br>

2. **End-to-end finetuning EfficientNetB0 classifier:** <br>
 <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model2_architecture.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model2_architecture.png">
  </a>
</p>

 <br>

3. **Finetuning separated classifier model:** <br>
 <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model3_architecture.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/model3_architecture.png">
  </a>
</p>

 <br>

## 5.1. Training the Model

To train the model, follow these steps:

1. Set up the phase parameter in the `config/config.yml` file for training model.
2. Select appropriate classification model.
3. Run the `main.py` script.
4. The script will read the images from the specified dataset directory and preprocess them.
5. The model will be trained using the preprocessed images and the specified parameters.
6. After training, the model will be saved in the specified directory.
7. The accuracy and loss values for training and validation stages are plotted.
8. Test process will start in turn and corresponding confusion matrix and evaluation metrics are visualized.


### Data Augmentation:
The AI system employs a robust preprocessing pipeline to enhance the quality and effectiveness of the training data before feeding it into the classification model. The preprocessing steps in the implementation include random rotation, random horizontal flip, resizing, and normalization, designed to augment and standardize the input images for optimal learning and classification performance.

1.	**Random Rotation**: The system incorporates Random Rotation as a data augmentation technique to introduce variability and increase the diversity of the training dataset. By randomly rotating the images within a specified range, the model becomes more robust to different orientations of the shoes, enabling it to learn invariant features and generalize better to unseen instances. This augmentation strategy enhances the model's ability to cope with variations in the orientation of shoes present in the dataset, ultimately improving its classification accuracy.
2.	**Random Horizontal Flip**: To further enhance the dataset's diversity and prevent overfitting, The AI system utilizes Random Horizontal Flip. This technique flips the images horizontally at random during training, creating additional variations while preserving the essential characteristics of the shoes. By introducing mirrored versions of the images, the model learns to recognize shoes irrespective of their orientation, leading to improved generalization and robustness in classifying shoe types accurately.
3.	**Resizing**: The images in the dataset are resized to a standardized format to ensure uniformity in input dimensions across all samples. Resizing facilitates the efficient processing of images by fixing their dimensions to a predefined size, enabling the model to handle inputs of consistent shapes. This step prepares the images for seamless integration into the neural network architecture, streamlining the training process and enhancing the model's ability to learn discriminative features without being constrained by varying image sizes.
4.	**Rescaling**: Before training, the pixel values of the resized images are scaled to a common range, typically ranging between 0 and 1 or -1 and 1. Scaling standardizes the pixel intensities across images, mitigating discrepancies in pixel values that could hinder the convergence of the model during training. By normalizing the input data, the system ensures that the neural network receives inputs in a consistent and optimized format, facilitating effective learning and improving the model's performance in accurately classifying shoe categories.

 <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/training_validation_performance.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/training_validation_performance.png">
  </a>
</p>

 <br>

## 5.2. Testing the Model

To test the model, follow these steps:

1. Set up the phase parameter in the `config/config.yml` file for testing model.
2. The script will load the pre-trained model from the specified directory.
3. Run the `main.py` script.
4. The model will be tested on the provided test images.
5. The predictions will be displayed along with the corresponding image.


## 5.3. Performance Metrics and Evaluation

When evaluating trained models, we aim to use accuracy, precision, recall, and the F1 score as key metrics to assess their performance. These evaluation metrics provide valuable insights into the effectiveness and accuracy of the trained models in classification tasks. We also use a confusion matrix to evaluate the performance of the classification model.

 <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/confusion_matrix.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/confusion_matrix.png">
  </a>
</p>

 <br>

  <br>

<p align="center">
  <a href="https://github.com/AminRj66/shoe-classification/blob/main/outputs/metrics.png" target="_blank">
    <img src="https://github.com/AminRj66/shoe-classification/blob/main/outputs/metrics.png">
  </a>
</p>

 <br>

# 6. Future Improvements

•	**Testing new classification architectures:** <br>
In the future, the implemented AI system can benefit from several enhancements to further improve its performance and flexibility. One key area of improvement involves exploring new classification architectures to enhance model accuracy and efficiency. By experimenting with state-of-the-art architectures such as Transformer-based models or ensemble methods, we can potentially achieve better results on complex datasets and tasks.

•	**Handling shared methods using parent classes:** <br>
Using more object-oriented programming (OOP) principles can enhance the maintainability and scalability of the AI system. Introducing a parent class to handle shared methods and functionalities for different classifiers can streamline code development and reduce redundancy across multiple models. This approach promotes code reusability, facilitates easier model extension or modification, and enhances overall code organization and readability.

•	**Employing design patterns:** <br>
Implementing design patterns (e.g., Factory, Strategy, Singleton) to address common programming challenges, enhance code readability, and facilitate future modifications and extensions.

•	**Performance optimization:** <br>
Analyzing and implementing optimizations such as model distillation, and architecture search to improve the speed, memory footprint, and accuracy of the AI model.
