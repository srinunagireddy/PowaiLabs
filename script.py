#!/home/bhushan/pip/tf/bin/python

"""
================================================================================
PROJECT: Horse v/s Human Image Classification
MODULE:  imageClassification.py
VERSION: 1.5
AUTHORS: Bhushan Kelshikar
URL: https://www.powailabs.com
================================================================================
ResNet Image Classification Pipeline
Supports ResNet50, ResNet101, ResNet152 (V1 and V2)

This script performs image classification (horse v/s human) using various 
ResNet architectures. It handles compressed archives, sets up logging, trains 
the model, and validates it. Validation now includes a default progress bar 
showing identification counts.

Variants : ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

Pipeline:
  Data loading → Augmentation → Model build → Phase-1 training (frozen base)
  → Phase-2 fine-tuning (unfrozen top layers) → Evaluation → CSV export

Usage examples:
  # Train all 6 models with default settings
  python imageClassification.py -i ./input

  # Train specific models, 20 epochs, batch size 16
  python imageClassification.py -i ./input -m resnet50,resnet101v2 -e 20 -b 16

  # Skip training; load weights from a directory
  python imageClassification.py -i ./input -m resnet50,resnet101v2 -e 20 -b 16

  # Use imagenet weights (default) and write results to custom CSV
  python imageClassification.py -i ./input --train -c results.csv

Note: This script assumes the following directory structure for input data:
  <input_path>/training/horse/
  <input_path>/training/human/
  <input_path>/testing/horse/
  <input_path>/testing/human/
The script will automatically create directories for saving models and weights:
  <input_path>/saved_models/
  <input_path>/saved_weights/
"""

""" Standard library imports for file handling, logging, argument parsing, and performance tracking """
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
import csv
import time
import logging
import argparse
import tracemalloc
from datetime import datetime

""" Importing NumPy for array manipulations and calculations """    
import numpy as np

""" Importing TensorFlow and Keras modules for model construction, training, and evaluation """
import tensorflow as tf

""" Importing Model class for functional API model construction """
from tensorflow.keras import Model

""" Importing layers for model construction - GlobalAveragePooling2D, Dense, Dropout """
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Dropout)

""" Importing Adam optimizer for model compilation """
from tensorflow.keras.optimizers import Adam

""" Importing callbacks for training - ModelCheckpoint, EarlyStopping, ReduceLROnPlateau """
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

""" Importing ImageDataGenerator for data augmentation and loading """
from tensorflow.keras.preprocessing.image import ImageDataGenerator

""" Importing ResNet model constructors for both V1 and V2 variants """
from tensorflow.keras.applications import (
    ResNet50,   ResNet101,   ResNet152,
    ResNet50V2, ResNet101V2, ResNet152V2,
)

""" Importing ResNet preprocess_input functions for both V1 and V2 variants """
from tensorflow.keras.applications.resnet    import preprocess_input as preprocess_v1
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_v2

""" importing sklearn metrics for evaluation - confusion matrix, classification report, accuracy, precision, recall, f1 score """
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)

# Constants
""" Standard ResNet input dimension for resizing and model feeding."""
IMG_SIZE = (224, 224)

""" IMG_EXTS defines the set of valid image file extensions that the script will recognize when loading data. """
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

""" Supported ResNet V1 and V2 architectures for model selection and initialization. """
ALL_MODELS = ["resnet50", "resnet101", "resnet152", "resnet50v2", "resnet101v2", "resnet152v2"]

""" MODEL_REGISTRY maps model names to their constructors and corresponding preprocessing functions. """
MODEL_REGISTRY = {
    "resnet50":    (ResNet50,    preprocess_v1),
    "resnet101":   (ResNet101,   preprocess_v1),
    "resnet152":   (ResNet152,   preprocess_v1),
    "resnet50v2":  (ResNet50V2,  preprocess_v2),
    "resnet101v2": (ResNet101V2, preprocess_v2),
    "resnet152v2": (ResNet152V2, preprocess_v2),
}

""" CSV_FIELDS defines the columns for the output CSV file that will store the results of the model training and evaluation. """
CSV_FIELDS = [
    "timestamp", "model", "batch_size", "epochs",
    "train_time_phase1_s", "peak_mem_train_phase1_mb",
    "train_time_phase2_s", "peak_mem_train_phase2_mb",
    "val_time_s",          "peak_mem_val_mb",
    "total_time_s",        "peak_mem_total_mb",
    "val_loss",            "val_accuracy",
    "accuracy",  "precision", "recall",   "f1",
    "test_horses", "horses_identified",
    "test_humans", "humans_identified",
    "top_score",   "top_index",  "label",
    "confusion_matrix",
    "total_params", "trainable_params",
]


def setup_logging(log_level: str, logfile: str) -> logging.Logger:
    """ Sets up logging to both a file and the console with a timestamped filename. """
    
    """ Get the current timestamp """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    """ Create the log file path by appending the timestamp to the base log file name. """
    logpath = f"{logfile}.{ts}"

    """ Define the log message format to include the timestamp, log level, and message. """
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    """ Get the logging level from the provided log_level string, defaulting to INFO if the string is not recognized. """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    """ Create a logger instance with the name 'ResNetClassifier' and set its logging level. """
    logger = logging.getLogger("ResNetClassifier")

    """ Set the logging level for the logger. """
    logger.setLevel(level)

    """ Create a file handler that writes log messages to the specified log file and set its formatter. """    
    fh = logging.FileHandler(logpath)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    """ Create a stream handler that outputs log messages to the console (stderr) and set its formatter. """
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    """ Log an initial message indicating that logging has been initialized and specify the log file path. """
    logger.info("Logging initialised → %s", logpath)

    """ Return the configured logger instance for use in the rest of the script. """
    return logger


def build_generators(input_path: str, preprocess_fn, batch_size: int, logger: logging.Logger):
    """ Builds training and validation data generators using Keras' ImageDataGenerator with specified preprocessing and augmentation. """
    
    """ Constructs the directory paths for training and testing datasets based on the provided input path. """
    train_dir = os.path.join(input_path, "training")
    test_dir  = os.path.join(input_path, "testing")

    """ Initializes the training data generator with specified preprocessing and augmentation parameters. """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    """ Initializes the validation data generator with only the specified preprocessing function, without augmentation. """
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    """ Creates an iterator for the training data that reads images from the training directory, applies the specified augmentations and preprocessing, and prepares batches of data for model training. """
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True,
    )

    """ Creates an iterator for the validation data that reads images from the testing directory, applies the specified preprocessing, and prepares batches of data for model evaluation. """
    val_gen = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    
    """ Logs the number of training and validation samples found by the generators. """
    logger.info("Training samples: %d | Validation samples: %d", train_gen.samples, val_gen.samples)
    
    """ Returns the configured training and validation data generators for use in model training and evaluation. """
    return train_gen, val_gen


def build_model(model_name: str, logger: logging.Logger, weights: str = "imagenet") -> tuple:
    """ Builds a Keras model based on the specified ResNet architecture, with options for using pretrained ImageNet weights. """
    
    """ Retrieves the model constructor and preprocessing function from the MODEL_REGISTRY based on the provided model name. """
    constructor, _ = MODEL_REGISTRY[model_name]
    
    """ 
    Initializes the base model using the constructor, specifying that the top 
    classification layer should be excluded, and optionally loading pretrained 
    weights. The input shape is set to match the expected dimensions for 
    ResNet models. 
    """
    base = constructor(
        include_top=False,
        weights=weights if weights == "imagenet" else None,
        input_shape=(*IMG_SIZE, 3),
    )
    
    """ 
    Freezes the base model layers to prevent them from being updated during 
    the initial training phase. This allows the model to leverage the pretrained 
    features from ImageNet while training only the newly added classification 
    layers. By setting trainable to False, we ensure that the weights of the 
    base model are not updated during backpropagation, which is crucial for 
    transfer learning when using pretrained models. This is typically done in 
    the first phase of training to allow the new layers to learn without 
    altering the pretrained features. After the initial training phase, we 
    can unfreeze some of the top layers of the base model for fine-tuning, 
    allowing the model to adapt the pretrained features to the specific task 
    at hand. For example, in the second phase of training, we might unfreeze 
    the top 30 layers of the base model to allow for fine-tuning while keeping 
    the lower layers frozen to retain the general features learned from ImageNet. 
    """
    base.trainable = False

    
    """ 
    Constructs the new classification head by adding a GlobalAveragePooling2D 
    layer to reduce the spatial dimensions, followed by a Dense layer with 
    ReLU activation for learning complex patterns, a Dropout layer for 
    regularization to prevent overfitting, and a final Dense layer with sigmoid 
    activation for binary classification. The GlobalAveragePooling2D layer 
    reduces the spatial dimensions of the feature maps output by the base model 
    to a single vector per feature map, which helps to reduce the number of 
    parameters and computational complexity while retaining important information. 
    """
    x = GlobalAveragePooling2D()(base.output)
    
    """ 
    The Dense layer with 256 units and ReLU activation serves as a fully 
    connected layer that learns to combine the features extracted by the base 
    model into more complex representations that are useful for the 
    classification task. The ReLU activation function introduces non-linearity, 
    allowing the model to learn more complex patterns in the data. 
    """
    x = Dense(256, activation="relu")(x)
    
    """ 
    The Dropout layer randomly sets a fraction of the input units to 0 at each 
    update during training time, which helps prevent overfitting by introducing 
    regularization. In this case, a dropout rate of 0.5 means that 50% of the 
    input units will be dropped out during training, which encourages the model 
    to learn more robust features that are not reliant on any single neuron. 
    """
    x = Dropout(0.5)(x)
    
    """ 
    The final output layer uses a sigmoid activation function to produce a 
    probability score between 0 and 1, indicating the likelihood of the input 
    image belonging to the positive class (e.g., human). 
    """
    out = Dense(1, activation="sigmoid")(x)
    
    """ 
    Constructs the final model by specifying the inputs as the base model's 
    input and the outputs as the output of the newly added classification 
    head. The model is named according to the specified model name for clarity 
    in logging and saving. 
    """
    model = Model(inputs=base.input, outputs=out, name=model_name)
    
    """ 
    Compiles the model using the Adam optimizer with a learning rate of 1e-3, 
    binary cross-entropy loss for binary classification, and accuracy as 
    the evaluation metric. The Adam optimizer is an adaptive learning rate 
    optimization algorithm that combines the benefits of both AdaGrad and 
    RMSProp, making it well-suited for training deep neural networks. The 
    binary cross-entropy loss function is appropriate for binary classification 
    tasks, as it measures the difference between the predicted probabilities 
    and the true binary labels. Accuracy is used as a metric to evaluate the 
    performance of the model during training and validation. By compiling the 
    model with these settings, we prepare it for training, allowing it to 
    optimize the weights based on the specified loss function and evaluate its 
    performance using the chosen metric. In the first phase of training, we 
    will keep the base model frozen and only train the newly added classification 
    head. After this initial phase, we can unfreeze some of the top layers of 
    the base model for fine-tuning, allowing the model to adapt the pretrained 
    features to our specific task while still retaining the general features 
    learned from ImageNet. In the second phase of training, we will typically 
    use a lower learning rate (e.g., 1e-5) to fine-tune the model, as we want 
    to make smaller updates to the weights of the base model to avoid overfitting 
    and ensure that we are not drastically altering the pretrained features. For 
    example, after the initial training phase, we might unfreeze the top 30 layers 
    of the base model and compile the model again with a lower learning rate for 
    fine-tuning. This allows the model to adapt the pretrained features to our 
    specific task while still retaining the general features learned from ImageNet.
    Overall, this function constructs a Keras model based on the specified ResNet 
    architecture, with options for using pretrained weights, and prepares it for 
    training by compiling it with the appropriate optimizer, loss function, and 
    evaluation metric. The function returns the constructed model and the base 
    model for potential fine-tuning in later phases. The base model is returned 
    separately to allow for easy access when we want to unfreeze layers for 
    fine-tuning in the second phase of training. By keeping the base model 
    separate, we can easily modify its trainable status and recompile the model 
    without having to reconstruct the entire model architecture. For example, 
    in the second phase of training, we can unfreeze the top 30 layers of the 
    base model and recompile the model with a lower learning rate for fine-tuning. 
    This allows us to adapt the pretrained features to our specific task while 
    still retaining the general features learned from ImageNet. By returning both 
    the constructed model and the base model, we provide flexibility for training 
    and fine-tuning in different phases of the training process. The function 
    also logs the model architecture summary, including the total number of 
    parameters and trainable parameters, which can be useful for understanding 
    the complexity of the model and for debugging purposes. Overall, this function 
    is responsible for building and compiling the Keras model based on the specified 
    ResNet architecture, with options for using pretrained weights, and preparing 
    it for training by compiling it with the appropriate settings. It also provides 
    logging of the model architecture and parameter counts for transparency and 
    debugging. 
    """
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    """ 
    Logs the model architecture summary, including the total number of 
    parameters and trainable parameters. 
    """
    model.summary(print_fn=lambda x: logger.info(x))
    
    """ Logs the total number of parameters and trainable parameters in the model. """
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    logger.info("Model %s - Total params: %d, Trainable params: %d", model_name, total_params, trainable_params)    
    
    """ Returns the constructed model and the base model for potential fine-tuning in later phases. """
    return model, base


def train_model(model, base_model, train_gen, val_gen,
                epochs: int, model_name: str, weights_dir: str,
                logger: logging.Logger) -> dict:
    """ 
    Trains the model in two phases: Phase 1 with a frozen backbone and 
    Phase 2 with the top 30 layers unfrozen for fine-tuning. 
    """
    """ 
    Constructs the checkpoint path for saving the best model weights during 
    training. The checkpoint will be saved in the specified weights directory 
    with a filename that includes the model name and a suffix indicating that 
    it is the best model based on validation loss. 
    """    
    ckpt_path = os.path.join(weights_dir, f"{model_name}_best.keras")
    os.makedirs(weights_dir, exist_ok=True)
    
    """ 
    Sets up callbacks for training, including ModelCheckpoint to save the 
    best model based on validation loss, EarlyStopping to stop training 
    if validation loss does not improve for a certain number of epochs, 
    and ReduceLROnPlateau to reduce the learning rate if validation 
    loss plateaus. 
    """
    callbacks = [
        ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss", verbose=0),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # Phase 1
    """ Logs the start of Phase 1 training with a frozen backbone. """
    """ Starts tracking memory usage with tracemalloc to measure the peak memory usage during training. """
    """ Records the start time for Phase 1 training. """
    logger.info("[%s] Phase 1 training (frozen backbone)...", model_name)
    
    
    """ Trains the model using the training generator, with the specified number of epochs, validation data, and callbacks. The verbose parameter is set to 1 to display progress during training. """
    """ 
    Starts tracking memory usage with tracemalloc to measure the peak memory 
    usage during training. This allows us to monitor the memory consumption 
    of the training process and identify any potential issues related to 
    memory usage. 
    """
    tracemalloc.start()
    
    """ Records the start time for Phase 1 training. """
    t0 = time.time()
    
    """ 
    The model is trained using the fit method, which takes the training generator, 
    number of epochs, validation data generator, and callbacks as arguments. 
    The callbacks will handle saving the best model weights, early stopping, and 
    learning rate reduction based on validation loss. 
    """
    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks, verbose=1)
    
    """ Records the end time for Phase 1 training. """
    t1 = time.time()
    
    """ 
    After training is complete, it retrieves the current memory usage and peak memory 
    usage from tracemalloc. The peak memory usage is then converted from bytes to 
    megabytes for easier interpretation. 
    """
    _, peak1 = tracemalloc.get_traced_memory()
    
    """ Stops tracking memory usage with tracemalloc after training is complete. """
    tracemalloc.stop()
    
    """ Calculates the time taken for Phase 1 training by subtracting the start time from the end time. """
    phase1_time = t1 - t0
    
    """ Converts the peak memory usage from bytes to megabytes for easier interpretation. """
    phase1_peak = peak1 / 1e6 

    # Phase 2
    """ 
    Logs the start of Phase 2 fine-tuning with the top 30 layers of the base model unfrozen. 
    """
    """ 
    Unfreezes the top 30 layers of the base model to allow them to be updated 
    during training. This allows the model to fine-tune the pretrained features 
    from ImageNet to better fit the specific task of horse vs human classification. 
    By unfreezing the top layers, we allow the model to adapt the higher-level 
    features learned from ImageNet to our specific dataset, while still keeping the 
    lower layers frozen to retain the general features learned from ImageNet. This 
    is a common approach in transfer learning, where we first train the model with a 
    frozen backbone to learn the new classification head, and then fine-tune the model 
    by unfreezing some of the top layers of the base model to adapt the pretrained 
    features to our specific task. 
    """
    logger.info("[%s] Phase 2 fine-tuning (top 30 layers unfrozen)...", model_name)
    
    """ 
    Unfreezes the top 30 layers of the base model to allow them to be updated 
    during training. This allows the model to fine-tune the pretrained 
    features from ImageNet to better fit the specific task of horse vs human 
    classification. By unfreezing the top layers, we allow the model to adapt 
    the higher-level features learned from ImageNet to our specific dataset, 
    while still keeping the lower layers frozen to retain the general features 
    learned from ImageNet. This is a common approach in transfer learning, 
    where we first train the model with a frozen backbone to learn the new 
    classification head, and then fine-tune the model by unfreezing some of the 
    top layers of the base model to adapt the pretrained features to our 
    specific task. 
    """
    for layer in base_model.layers[-30:]:
        
        """ 
        Unfreezes the layer by setting its trainable attribute to True, allowing 
        it to be updated during training. This is done for the top 30 layers of 
        the base model to allow for fine-tuning while still retaining the general 
        features learned from ImageNet in the lower layers. By unfreezing these 
        layers, we enable the model to adapt the pretrained features to our 
        specific task of horse vs human classification, which can lead to 
        improved performance on our dataset. 
        """
        layer.trainable = True

    """ 
    Recompiles the model with a lower learning rate for fine-tuning. This is 
    important because we want to make smaller updates to the weights of the 
    base model during fine-tuning to avoid overfitting and ensure that we are 
    not drastically altering the pretrained features. By using a lower 
    learning rate (e.g., 1e-5), we allow the model to adapt the pretrained 
    features to our specific task while still retaining the general features 
    learned from ImageNet. 
    """
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])
    
    """ 
    Starts tracking memory usage with tracemalloc to measure the peak memory 
    usage during fine-tuning. This allows us to monitor the memory consumption 
    of the fine-tuning process and identify any potential issues related to 
    memory usage. 
    """
    tracemalloc.start()
    
    """ Records the start time for Phase 2 fine-tuning. """
    t2 = time.time()
    
    """ 
    The model is trained again using the fit method, with the same training 
    generator, number of epochs (or a reduced number of epochs for fine-tuning), 
    validation data generator, and callbacks as before. The callbacks will 
    continue to handle saving the best model weights, early stopping, and 
    learning rate reduction based on validation loss during this fine-tuning phase. 
    """
    model.fit(train_gen, epochs=max(epochs // 2, 5), validation_data=val_gen, callbacks=callbacks, verbose=1)
    
    """ Records the end time for Phase 2 fine-tuning. """
    t3 = time.time()
    
    """ 
    After fine-tuning is complete, it retrieves the current memory usage and peak 
    memory usage from tracemalloc. The peak memory usage is then converted from 
    bytes to megabytes for easier interpretation. 
    """
    _, peak2 = tracemalloc.get_traced_memory()
    
    """ Stops tracking memory usage with tracemalloc after fine-tuning is complete. """
    tracemalloc.stop()

    """ Calculates the time taken for Phase 2 fine-tuning. """
    phase2_time = t3 - t2
    
    """ Converts the peak memory usage from bytes to megabytes for easier interpretation. """
    phase2_peak = peak2 / 1e6
    
    """ 
    Returns a dictionary containing the training time and peak memory usage for both 
    Phase 1 and Phase 2, as well as the checkpoint path where the best model weights 
    were saved. The training times and peak memory usages are rounded to 2 decimal 
    places for easier readability. 
    """
    return {
        "train_time_phase1_s":       round(phase1_time, 2),
        "peak_mem_train_phase1_mb":  round(phase1_peak, 2),
        "train_time_phase2_s":       round(phase2_time, 2),
        "peak_mem_train_phase2_mb":  round(phase2_peak, 2),
        "ckpt_path":                 ckpt_path,
    }


def validate_model(model, val_gen, logger: logging.Logger) -> dict:
    """ 
    Validates the model on the validation set, measuring validation time, 
    peak memory usage, and various performance metrics including accuracy, 
    precision, recall, F1 score, and confusion matrix. 
    """
    logger.info("Validating...")
    
    """ 
    Starts tracking memory usage with tracemalloc to measure the peak memory 
    usage during validation. This allows us to monitor the memory consumption 
    of the validation process and identify any potential issues related to 
    memory usage. 
    """
    tracemalloc.start()
    
    """ Records the start time for validation. """
    t0 = time.time()
    
    """ 
    Evaluates the model on the validation generator to obtain the validation loss 
    and accuracy. The verbose parameter is set to 1 to display progress during evaluation. 
    """
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    
    """ 
    Resets the validation generator to ensure that it starts from the beginning of the 
    dataset for any subsequent evaluations or predictions. This is important because 
    the generator maintains an internal state, and after evaluation, it may have 
    iterated through the dataset. By resetting it, we ensure that any future calls to 
    predict or evaluate will start from the beginning of the validation dataset. 
    """
    val_gen.reset()
    
    """ 
    Predicts probabilities for the validation set using the model's predict method. 
    The predictions are flattened to a 1D array for easier processing. 
    """
    preds_prob = model.predict(val_gen, verbose=1).flatten()
    
    """ Records the end time for validation. """
    t1 = time.time()
    
    """ 
    After validation is complete, it retrieves the current memory usage and 
    peak memory usage from tracemalloc. The peak memory usage is then converted 
    from bytes to megabytes for easier interpretation. 
    """
    _, peak = tracemalloc.get_traced_memory()
    
    """ Stops tracking memory usage with tracemalloc after validation is complete. """
    tracemalloc.stop()
    
    """ Calculates the time taken for validation by subtracting the start time from the end time. """
    val_time = round(t1 - t0, 2)
    
    """ Converts the peak memory usage from bytes to megabytes for easier interpretation. """
    val_peak = round(peak / 1e6, 2)
    
    """
    The predicted probabilities are converted to binary labels using a threshold of 0.5. 
    If the predicted probability is greater than or equal to 0.5, it is classified as 
    the positive class (e.g., human); otherwise it is classified as the negative class 
    (e.g., horse). This conversion is necessary for calculating performance metrics 
    such as accuracy, precision, recall, and F1 score, which require binary labels for 
    comparison with the true labels. By applying this threshold, we can evaluate how 
    well the model is performing in correctly classifying the validation samples as 
    either horses or humans. 
    """
    preds_bin = (preds_prob >= 0.5).astype(int)
    
    """ 
    The true labels from the validation generator are retrieved to determine 
    the actual class labels for the validation samples. This is important for 
    calculating performance metrics such as accuracy, precision, recall, and 
    F1 score by comparing the predicted binary labels with the true labels. 
    By obtaining the true labels, we can evaluate how well the model is 
    performing in correctly classifying the validation samples as either 
    horses or humans. 
    """
    true_labels = val_gen.classes
    
    """ 
    The class indices from the validation generator are retrieved to determine 
    the mapping of class labels to their corresponding indices. This is important 
    for calculating performance metrics specific to each class (e.g., horses and humans) 
    based on their assigned indices in the dataset. By obtaining the class indices, 
    we can accurately calculate performance metrics for each class. 
    """
    idx = val_gen.class_indices

    """ 
    The horse index is retrieved from the class indices of the validation generator. 
    This index corresponds to the label assigned to horse images in the dataset. 
    By obtaining this index, we can accurately calculate performance metrics specific 
    to horse images, such as the number of horses correctly identified and the number 
    of test horses in the validation set. This information is crucial for evaluating 
    the model's performance in correctly classifying horse images. 
    """
    h_idx = idx.get("horse", 0)
    
    """ 
    The human index is retrieved from the class indices of the validation 
    generator. This index corresponds to the label assigned to human images 
    in the dataset. By obtaining this index, we can accurately calculate 
    performance metrics specific to human images, such as the number of 
    humans correctly identified and the number of test humans in the 
    validation set. This information is crucial for evaluating the model's 
    performance in correctly classifying human images. 
    """
    hu_idx = idx.get("human", 1)

    """ 
    The number of test horses is calculated by summing the instances where 
    the true labels match the horse index (h_idx). This gives us the count 
    of actual horse images in the validation set, which is important for 
    evaluating the model's performance in correctly identifying horses. 
    """
    test_horses = int(np.sum(true_labels == h_idx))
    
    """ 
    The number of horses correctly identified is calculated by summing the 
    instances where the predicted binary labels match the horse index (h_idx) 
    and the true labels also match the horse index (h_idx). This gives us 
    the count of true positives for horses, which indicates how many horse 
    images were correctly classified as horses by the model. 
    """
    horses_id = int(np.sum((preds_bin == h_idx) & (true_labels == h_idx)))
    
    """ 
    The number of test humans is calculated by summing the instances where 
    the true labels match the human index (hu_idx). This gives us the count 
    of actual human images in the validation set, which is important for 
    evaluating the model's performance in correctly identifying humans. 
    """
    test_humans = int(np.sum(true_labels == hu_idx))
    
    """ 
    The number of humans correctly identified is calculated by summing the 
    instances where the predicted binary labels match the human index (hu_idx) 
    and the true labels also match the human index (hu_idx). This gives us 
    the count of true positives for humans, which indicates how many human 
    images were correctly classified as humans by the model. 
    """
    humans_id = int(np.sum((preds_bin == hu_idx) & (true_labels == hu_idx)))

    """ 
    The accuracy score is calculated using the true labels and the predicted 
    binary labels. The accuracy score is a measure of a model's overall 
    correctness, indicating the proportion of correct predictions (both true 
    positives and true negatives) out of all predictions made by the model. 
    It is calculated as the ratio of correctly predicted instances to the total 
    number of instances in the validation set, and it ranges from 0 to 1, where 
    1 indicates perfect accuracy. In this case, the accuracy score is rounded 
    to 4 decimal places for precision. 
    """
    acc = round(accuracy_score(true_labels, preds_bin), 4)
    
    """ 
    The precision score is calculated using the true labels and the predicted 
    binary labels. The precision score is a measure of a model's ability to 
    correctly identify positive instances (e.g., correctly identifying humans 
    in this case) out of all instances that were predicted as positive by the 
    model. It is calculated as the ratio of true positives to the sum of true 
    positives and false positives, and it ranges from 0 to 1, where 1 indicates 
    perfect precision. In this case, the precision score is rounded to 4 decimal 
    places for precision. The zero_division parameter is set to 0 to handle cases 
    where there are no positive predictions or no true positives, preventing 
    division by zero errors. 
    """    
    prec = round(precision_score(true_labels, preds_bin, zero_division=0), 4)
    
    """ 
    The recall score is calculated using the true labels and the predicted 
    binary labels. The recall score is a measure of a model's ability to 
    correctly identify positive instances (e.g., correctly identifying humans 
    in this case) out of all actual positive instances in the dataset. It is 
    calculated as the ratio of true positives to the sum of true positives and 
    false negatives, and it ranges from 0 to 1, where 1 indicates perfect recall. 
    In this case, the recall score is rounded to 4 decimal places for precision. 
    The zero_division parameter is set to 0 to handle cases where there are no 
    positive predictions or no true positives, preventing division by zero errors. 
    """
    rec = round(recall_score(true_labels, preds_bin, zero_division=0), 4)
    
    """ 
    The F1 score is calculated using the true labels and the predicted binary 
    labels. The F1 score is a measure of a model's accuracy that considers 
    both precision and recall, providing a balance between the two. It is 
    calculated as the harmonic mean of precision and recall, and it ranges from 
    0 to 1, where 1 indicates perfect precision and recall. In this case, the 
    F1 score is rounded to 4 decimal places for precision. The zero_division 
    parameter is set to 0 to handle cases where there are no positive predictions 
    or no true positives, preventing division by zero errors. 
    """
    f1 = round(f1_score(true_labels, preds_bin, zero_division=0), 4)
    
    """ 
    The confusion matrix is computed using the true labels and the predicted 
    binary labels. The confusion matrix is then converted to a list for easier 
    storage in the CSV file. The confusion matrix provides a summary of the 
    model's performance by showing the counts of true positives, true negatives, 
    false positives, and false negatives, which can be useful for understanding 
    the types of errors the model is making and for calculating various 
    performance metrics. 
    """
    cm = confusion_matrix(true_labels, preds_bin).tolist()

    """ 
    The top index is determined by finding the index of the highest predicted 
    probability from the model's predictions on the validation set. This index 
    corresponds to the sample in the validation set that the model is most 
    confident about in terms of its prediction. By identifying this top index, 
    we can analyze the model's most confident prediction and understand how well 
    it is performing on that particular sample. This information can be useful 
    for setting thresholds for classification and for gaining insights into the 
    model's behavior on the validation set. 
    """
    top_idx = int(np.argmax(preds_prob))
    
    """ 
    The top score is the highest predicted probability from the model's 
    predictions on the validation set. It is rounded to 4 decimal places for 
    precision. This score represents the model's confidence in its most 
    confident prediction, which can be useful for understanding how well the 
    model is performing and for setting thresholds for classification. 
    """
    top_score = round(float(preds_prob[top_idx]), 4)
    
    """ 
    The predicted label for the top score is determined based on a threshold 
    of 0.5. If the top score is greater than or equal to 0.5, the predicted 
    label is "human"; otherwise the predicted label is "horse". This threshold 
    is commonly used in binary classification tasks to determine the class 
    label based on the predicted probability. By assigning a label based on 
    this threshold, we can interpret the model's predictions in terms of the 
    actual classes (horses and humans) and evaluate its performance accordingly. 
    """
    label = "human" if top_score >= 0.5 else "horse"

    """ 
    Returns a dictionary containing the validation time, peak memory usage, 
    validation loss, validation accuracy, and various performance metrics 
    including accuracy, precision, recall, F1 score, number of test horses 
    and humans, number of horses and humans correctly identified, top 
    prediction score and index, predicted label for the top score, and the 
    confusion matrix. The validation time and peak memory usage are rounded to 
    2 decimal places for easier readability. The performance metrics are 
    rounded to 4 decimal places for precision. The confusion matrix is 
    converted to a list for easier storage in the CSV file.
    """
    return {
        "val_time_s":       val_time,
        "peak_mem_val_mb":  val_peak,
        "val_loss":         round(float(val_loss), 4),
        "val_accuracy":     round(float(val_acc), 4),
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "test_horses":      test_horses,
        "horses_identified":horses_id,
        "test_humans":      test_humans,
        "humans_identified":humans_id,
        "top_score":        top_score,
        "top_index":        top_idx,
        "label":            label,
        "confusion_matrix": str(cm),
    }


def append_csv(csvfile: str, row: dict) -> None:
    """ 
    Appends a row of data to a CSV file, creating the file with headers if it 
    does not already exist. 
    """
    """ 
    Defines the fieldnames for the CSV file, which correspond to the keys in 
    the row dictionary. These fieldnames will be used as the headers for the 
    CSV file and should match the keys in the row dictionary to ensure that the 
    data is correctly written to the file. 
    """
    """
    CSV_FIELDS = [
        "timestamp", "model", "batch_size", "epochs", "total_time_s", "peak_mem_total_mb",
        "total_params", "trainable_params",
        "train_time_phase1_s", "peak_mem_train_phase1_mb",
        "train_time_phase2_s", "peak_mem_train_phase2_mb",
        "val_time_s", "peak_mem_val_mb", "val_loss", "val_accuracy",
        "accuracy", "precision", "recall", "f1",
        "test_horses", "horses_identified",
        "test_humans", "humans_identified",
        "top_score", "top_index", "label",
        "confusion_matrix",
    ]
    """
    
    """ 
    Checks if the CSV file already exists. If it does not exist, it will be 
    created and the headers will be written to the file. If it already exists, 
    the new row will be appended without writing the headers again. This ensures 
    that the CSV file is properly formatted with headers and that new data is 
    added correctly without duplicating headers. 
    """  
    file_exists = os.path.isfile(csvfile)
    with open(csvfile, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        
        """ 
        If the file does not exist, write the header row to the CSV file. 
        This is done only once when the file is created to ensure that 
        the headers are included at the top of the file. If the file 
        already exists, this step is skipped to avoid writing duplicate headers. 
        """
        if not file_exists:
            writer.writeheader()
        
        """ 
        Writes the provided row of data to the CSV file. The row should be a 
        dictionary where the keys correspond to the fieldnames defined earlier. 
        The extrasaction="ignore" parameter ensures that any keys in the row 
        that are not defined in the fieldnames will be ignored and not written 
        to the file, preventing errors and ensuring that only the specified 
        fields are included in the CSV output. 
        """
        writer.writerow(row)


def run_pipeline(args, logger: logging.Logger) -> None:
    """ 
    Runs the entire training and validation pipeline for the specified models, 
    including building the model, training in two phases, validating, and saving 
    results to a CSV file. 
    """
    
    """ 
    Parses the list of models to run from the command-line arguments, ensuring that 
    they are valid and exist in the MODEL_REGISTRY. If any invalid model names are 
    provided, an error is logged and the program exits. This ensures that only supported 
    models are trained and evaluated in the pipeline. 
    """
    models_to_run = [m.strip().lower() for m in args.models.split(",")]
    
    """ 
    Checks for any invalid model names that are not present in the MODEL_REGISTRY. 
    If any invalid models are found, an error message is logged listing the unknown 
    models and the valid options, and the program exits with a non-zero status code 
    to indicate an error. This validation step ensures that the user provides valid 
    model names and prevents the pipeline from running with unsupported models. """
    invalid = [m for m in models_to_run if m not in MODEL_REGISTRY]

    if invalid:
        """ 
        Logs an error message listing the unknown model names that were provided 
        in the command-line arguments, along with the valid model options from 
        the MODEL_REGISTRY. This helps the user understand which model names are 
        not recognized and what the valid options are for running the pipeline. 
        After logging the error, the program exits with a non-zero status code to 
        indicate that an error occurred due to invalid input. 
        """
        logger.error("Unknown model(s): %s. Valid: %s", invalid, ALL_MODELS)
        
        """ 
        Exits the program with a non-zero status code to indicate that an error 
        occurred due to invalid model names provided in the command-line arguments. 
        This prevents the pipeline from running with unsupported models and allows 
        the user to correct their input before trying again. 
        """
        sys.exit(1)

    """ 
    Determines whether to use pretrained ImageNet weights based on the 
    command-line arguments. If --train is 'imagenet' (the default), pretrained 
    ImageNet weights are used for transfer learning. If --train is 'from-scratch', 
    the backbone is randomly initialised and trained without any pretrained weights. 
    If --notrain is provided, training is skipped entirely and weights are loaded 
    from the specified directory. 
    """
    use_imagenet = (args.train == "imagenet")
    
    """ 
    If the --notrain flag is provided with a directory, the pipeline will skip 
    the training phase and instead load the best model weights from the 
    specified directory for validation. This allows users to evaluate the 
    performance of a previously trained model without having to retrain it, 
    which can save time and computational resources when they are only interested 
    in validating the model's performance on the validation set. 
    """
    notrain_dir = args.notrain
    
    """ 
    Records the current timestamp in a human-readable format to be included 
    in the results saved to the CSV file. This timestamp can be useful for 
    tracking when the training and validation were performed, especially when 
    running multiple experiments or comparing results over time. 
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    """ 
    Records the start time for the entire pipeline to calculate the total time 
    taken for training and validation across all models. This allows us to 
    measure the overall performance of the pipeline and compare it across 
    different runs or configurations. 
    """
    grand_start = time.time()

    """ 
    Iterates over the list of models to run, performing training and validation 
    for each model sequentially. For each model, it logs the model name, builds 
    the data generators, constructs the model architecture, trains the model 
    (if not skipping training), validates the model, saves the trained model, 
    and appends the results to the CSV file. This loop allows us to efficiently 
    run multiple models in a single execution of the pipeline and compare their 
    performance based on the results saved in the CSV file. 
    """
    for model_name in models_to_run:
        logger.info("=" * 60)
        logger.info("Model: %s", model_name.upper())
        logger.info("=" * 60)

        """ 
        Retrieves the preprocessing function for the specified model from the 
        MODEL_REGISTRY. The preprocessing function is used to prepare the input 
        images in a way that is compatible with the pretrained weights of the 
        model (if using pretrained weights) and to ensure that the input data is 
        properly normalized and formatted for training and validation. By obtaining 
        the appropriate preprocessing function, we can ensure that the input data 
        is correctly processed before being fed into the model, which can improve 
        training stability and performance. 
        """
        _, preprocess_fn = MODEL_REGISTRY[model_name]
        
        
        """ 
        Builds the training and validation data generators using the specified input 
        directory, preprocessing function, batch size, and logger. The data generators 
        will handle loading and preprocessing the images from the training and validation 
        directories, providing batches of data to the model during training and validation. 
        By using data generators, we can efficiently manage memory usage and ensure that 
        the data is properly preprocessed on-the-fly during training and validation. 
        """
        train_gen, val_gen = build_generators(args.input, preprocess_fn, args.batch_size, logger)

        """ 
        Determines the weight initialisation mode: 'imagenet' loads pretrained 
        ImageNet weights for transfer learning; 'from-scratch' uses random 
        initialisation. When --notrain is supplied, this value is not used for 
        training but still informs the model constructor call. 
        """
        weights = "imagenet" if use_imagenet else None
        
        """ 
        Builds the model architecture for the specified model name, using the 
        provided logger and weights option. The build_model function constructs 
        the Keras model based on the specified ResNet architecture, with options 
        for using pretrained weights, and prepares it for training by compiling 
        it with the appropriate optimizer, loss function, and evaluation metric. 
        The function returns the constructed model and the base model for 
        potential fine-tuning in later phases. By building the model architecture, 
        we can prepare it for training and validation in the subsequent steps of 
        the pipeline. 
        """
        model, base_model = build_model(model_name, logger, weights=weights)

        """ 
        Calculates the total number of parameters in the model using the 
        count_params method. This gives us an understanding of the size 
        and complexity of the model, which can be useful for debugging 
        purposes and for making informed decisions about model architecture 
        and training strategies. By logging the total parameter count, we can 
        gain insights into how many parameters are present in the model, which 
        can help us understand its capacity and potential performance on the 
        task at hand. 
        """ 
        total_params = model.count_params()
        
        """ 
        Logs the total number of parameters in the model. This information 
        can be useful for understanding the complexity of the model and for 
        debugging purposes. By logging the total parameter count, we can gain 
        insights into the size of the model and how many parameters it contains, 
        which can help us make informed decisions about model architecture and 
        training strategies. 
        """
        logger.info("Total params: %d", total_params)
        
        """ 
        Calculates the total number of trainable parameters in the model by summing 
        the sizes of all trainable weights. This gives us an understanding of how 
        many parameters will be updated during training, which can be useful for 
        understanding the complexity of the model and for debugging purposes. By 
        logging the trainable parameter count, we can gain insights into how many 
        parameters are being optimized during training, which can help us make 
        informed decisions about model architecture and training strategies. 
        """   
        trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
        
        """ 
        Logs the total number of parameters and trainable parameters in the model. 
        This information can be useful for understanding the complexity of the model 
        and for debugging purposes. By logging these parameter counts, we can gain 
        insights into the size of the model and how many parameters are being updated 
        during training, which can help us make informed decisions about model architecture 
        and training strategies. 
        """
        logger.info("Total params: %d, Trainable params: %d", total_params, trainable_params)

        if notrain_dir:
            phase2_path = os.path.join(notrain_dir, f"{model_name}_best.keras")
            if os.path.exists(phase2_path):
                logger.info("Loading weights from %s", phase2_path)
                model.load_weights(phase2_path)
            else:
                logger.warning("No saved weights found for %s", model_name)
            
            train_metrics = {
                "train_time_phase1_s": 0, "peak_mem_train_phase1_mb": 0,
                "train_time_phase2_s": 0, "peak_mem_train_phase2_mb": 0,
            }
        else:
            train_metrics = train_model(
                model, base_model, train_gen, val_gen,
                args.epochs, model_name,
                weights_dir=os.path.join(args.input, "saved_weights"),
                logger=logger,
            )

        """ 
        Validates the model on the validation set, measuring validation time, 
        peak memory usage, and various performance metrics including accuracy, 
        precision, recall, F1 score, and confusion matrix. The validate_model 
        function evaluates the model's performance on the validation set and 
        returns a dictionary containing the validation metrics, which can be 
        used for analysis and comparison of different models. By validating the 
        model, we can assess its performance on unseen data and gain insights 
        into its strengths and weaknesses in classifying horses vs humans. 
        """
        val_metrics = validate_model(model, val_gen, logger)

        """ 
        Saves the trained model to a directory named "saved_models" 
        within the input directory, with a subdirectory for each model 
        name. This allows us to keep track of the trained models and their 
        corresponding weights for future use or analysis. By saving the model, 
        we can easily load it later for inference or further fine-tuning 
        without having to retrain it from scratch. 
        """
        save_dir = os.path.join(args.input, "saved_models", model_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{model_name}.keras")
        model.save(save_path)
        logger.info("Model saved to %s", save_path)

        """ Collects all peak memory usage values for reporting. """
        all_peaks = [
            train_metrics.get("peak_mem_train_phase1_mb", 0),
            train_metrics.get("peak_mem_train_phase2_mb", 0),
            val_metrics["peak_mem_val_mb"],
        ]
        row = {
            "timestamp":        timestamp,
            "model":            model_name,
            "batch_size":       args.batch_size,
            "epochs":           args.epochs,
            "total_time_s":     round(time.time() - grand_start, 2),
            "peak_mem_total_mb":round(max(all_peaks), 2),
            "total_params":     total_params,
            "trainable_params": trainable_params,
            **train_metrics,
            **val_metrics,
        }
        append_csv(args.csvfile, row)

    logger.info("All models complete.")


def main():
    parser = argparse.ArgumentParser(
        prog="imageClassification.py",
        description="Horse vs Human binary classifier using ResNet variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Optional input path with PWD default
    default_path = os.path.join(os.environ.get('PWD', os.getcwd()), 'models/')
    parser.add_argument(
        "-i", "--input",
        metavar="INPUT_DATA_DIR", 
        default=default_path,
        help="Root input data directory containing training/ and testing/. \
        Default: {}/testing {}/training".format(default_path, default_path)
    )

    """ 
    Define models to run as a comma-separated list. The default is to run all 
    models defined in the MODEL_REGISTRY. This argument allows users to specify 
    which models they want to train and validate, providing flexibility in 
    benchmarking different architectures without having to modify the code. The 
    specified models must be valid keys in the MODEL_REGISTRY, otherwise an 
    error will be logged and the program will exit. 
    """ 
    parser.add_argument(
        "-m", "--models",
        metavar="MODEL_LIST", 
        default=",".join(ALL_MODELS), 
        help="List of comma separated models to run"
    )

    """ 
    Define the path to the output CSV file where benchmarking results will be 
    saved. The default is "result-benchmark.csv". This argument allows users 
    to specify a custom location for the CSV output, which can be useful for 
    organizing results from different runs or for saving results in a specific 
    directory. The CSV file will contain detailed metrics and information about 
    the training and validation process for each model. 
    """
    parser.add_argument(
        "-c", "--csvfile",
        metavar="CSV_FILE", 
        default="result-benchmark.csv", 
        help="Path to output CSV file"
    )

    """ 
    Define the batch size for training and validation. The default is 32. This 
    argument allows users to specify the batch size, which can impact the 
    training time and memory usage of the model. A larger batch size may lead 
    to faster training but higher memory usage, while a smaller batch size may 
    reduce memory requirements but increase training time. The specified batch 
    size will be used when building the data generators for both training and 
    validation. 
    """
    parser.add_argument(
        "-b", "--batch-size",
        metavar="N", 
        type=int, 
        default=32, 
        dest="batch_size", 
        help="Batch size"
    )

    """ 
    Define the number of training epochs for Phase 1. The default is 5. This 
    argument allows users to specify how many epochs to train the model during 
    the initial phase with a frozen backbone. The number of epochs for Phase 2 
    fine-tuning will be set to half of this value (or at least 5) to allow for 
    sufficient training without overfitting. The specified number of epochs 
    will impact the training time and potentially the performance of the model, 
    as more epochs may lead to better convergence but also increased training 
    time. 
    """    
    parser.add_argument(
        "-e", "--epochs",
        metavar="N", 
        type=int, 
        default=5, 
        help="Epochs for Phase 1. For Phase 2 fine-tuning, it will use half \
        of this value or a minimum of 5 epochs."
    )

    """ 
    --log-level argument defines the level of information to add to the
        logfile. Default: INFO
    """
    parser.add_argument(
        "-L", "--log-level",
        metavar="LOG_LEVEL", 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    """ 
    --logfile argument allows users to specify a base name for the logfile 
        where training and validation logs will be saved. By default, it is 
        set to "resnet-regression.log". The timestamp will be automatically 
        appended to the logfile name to ensure that each run of the pipeline 
        generates a unique logfile. This allows users to keep track of logs 
        from different runs and easily identify them based on the timestamp 
        in the filename. By providing a customizable logfile name, users can 
        organize their logs in a way that suits their needs and makes it easier 
        to analyze the results of different runs of the pipeline. 
    """
    parser.add_argument(
        "-l", "--logfile", 
        metavar="LOG_FILE", 
        default="resnet-regression.log", 
        help="Base name for the logfile. Default: resnet-regression.log \
        (timestamp appended automatically)",
    )

    """ 
    --train accepts 'imagenet' (use pretrained ImageNet weights, default) 
        or 'from-scratch' (random initialisation, no pretrained weights). 
    --notrain skips training entirely and loads saved weights from WEIGHTS_DIR. 
    --train and --notrain are mutually exclusive. 
    """     
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "-t", "--train",
        metavar="MODE",
        choices=["imagenet", "from-scratch"],
        default="imagenet",
        help=(
            "Training mode. "
            "'imagenet'     - initialise backbone with pretrained ImageNet \
                weights (default). "
            "'from-scratch' - initialise backbone with random weights, \
                no transfer learning. "
            "Mutually exclusive with --notrain."
        ),
    )

    train_group.add_argument(
        "-n", "--notrain",
        metavar="WEIGHTS_DIR",
        default=None,
        help=(
            "Skip training and load previously saved weights from WEIGHTS_DIR. "
            "The directory must exist and contain <model>_best.keras files. "
            "Mutually exclusive with --train."
        ),
    )

    """ Parses and validates the command-line arguments """ 
    args = parser.parse_args()

    """ 
    Validates the --notrain directory if provided. If --notrain is specified,
    it checks whether the provided weights directory exists. If the  directory
    does not exist, it raises a parser error with a message indicating that 
    the specified directory does not exist. This validation step ensures that 
    when users choose to skip training and load weights from a directory, they  
    provide a valid directory path that contains the necessary saved weights 
    for the models they intend to validate. 
    """
    if args.notrain is not None:
        if not os.path.isdir(args.notrain):
            parser.error(f"--notrain directory does not exist: {args.notrain}")

    """ Sets up logging based on the specified log level and logfile name """
    logger = setup_logging(args.log_level, args.logfile)

    """ 
    Runs the entire training and validation pipeline for the specified models, 
    including building the model, training in two phases, validating, and 
    saving results to a CSV file. The run_pipeline function orchestrates the
    entire process of training and validating the models based on the provided  
    command-line arguments and logs important information throughout the
    execution. By running the pipeline, we can efficiently train and evaluate 
    multiple models, compare their performance, and save the results for
    further analysis. 
    """
    run_pipeline(args, logger)

""" 
The main entry point of the script. When the script is executed, the main()
function is called, which sets up argument parsing, logging, and runs the  
training and validation pipeline for the specified models. This allows us 
to execute the entire process of training and validating the models by 
simply running the script with the appropriate command-line arguments. 
"""

if __name__ == "__main__":
    main()