import pandas
import skimage.io
import os
import tqdm
import glob
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import RandomRotation, RandomZoom
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import applications
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, AUC, Recall
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.densenet import DenseNet169
import copy
import warnings
warnings.filterwarnings('ignore')
import cv2
from tensorflow.keras.utils import load_img, img_to_array
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import splitfolders

input_folder = r"C:\Kaggle\Input\augmented-alzheimer-mri-dataset-v2\Alzheimer_Dataset\test"

output_folder = r"C:\Kaggle\Output"

train_ratio=0.8
validation_ratio=0.1
test_ratio=0.1
splitfolders.ratio(input_folder, output_folder, seed=42,
                   ratio=(train_ratio,
                          validation_ratio,
                          test_ratio))

from keras.src.legacy.preprocessing.image import ImageDataGenerator

BATCH_SIZE=16
IMG_SIZE=(128,128)
SEED=1345

train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0,
                                zoom_range=0.2)

validation_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)


# Defining directories for train,validation,test

train_dir = r"\Alzheimer_Dataset\train"
validation_dir = r"\Alzheimer_Dataset\val"
test_dir = r"\Alzheimer_Dataset\test"



# Defining generatores for train,validation,test

train_generator = train_datagen.flow_from_directory(
    train_dir,
        target_size=(128, 128),
        shuffle=True,
        seed = SEED,
        batch_size=64,
        class_mode ='categorical',
)

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        seed = SEED,
        shuffle=True,
        batch_size=64,
        class_mode ='categorical',)

# Define generator for test set using flow_from_directory
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        shuffle=True,
        seed = SEED,
        batch_size =64,
        class_mode ='categorical',
)

class_names = list(train_generator.class_indices.keys())
print(class_names)


plt.figure(figsize=(12,12))

for images,labels in train_generator:
#     print(images)
#     print(len(labels))
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i])
        print(images[i].shape)
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis("off")
    break

def data_augmentar():
    data_augmentation = Sequential()
    data_augmentation.add(RandomRotation(factor=(-0.15, 0.15)))
    data_augmentation.add(RandomZoom((-0.3, -0.1)))
    return data_augmentation

data_augmentation = data_augmentar()
assert(data_augmentation.layers[0].name.startswith('random_rotation'))
assert(data_augmentation.layers[0].factor == (-0.15, 0.15))
assert(data_augmentation.layers[1].name.startswith('random_zoom'))
assert(data_augmentation.layers[1].height_factor == (-0.3, -0.1))

class_count=dict()

for i in class_names:
    class_count[i] = len(os.listdir(input_folder+'/'+i))

plt.figure(figsize=(10,4))
plt.bar(class_count.keys(),class_count.values())

plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Visualization of Data Imbalance')
plt.grid(True)
plt.show()

print("Completion check")

total_samples=sum(class_count.values())

print("Completed")

for i in range(4):
    class_weight = round(total_samples / (4 * list(class_count.values())[i]), 2)
    print(f'Weight for class \"{class_names[i]}\" : {class_weight}')


print("Done Test 1")

from keras import layers
import matplotlib.image as img

IMG_HEIGHT = 128
IMG_WIDTH = 128

model = keras.models.Sequential()
model.add(keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)))
model.add(keras.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

print("Done Test 2")

model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Dropout(0.20))

model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128,activation="relu",kernel_initializer="he_normal"))
model.add(keras.layers.Dense(64,"relu"))
model.add(keras.layers.Dense(4,"softmax"))

input_shape = (128,128, 3)

print("Done test 3")

train_features = model.predict(train_generator, steps=len(train_generator), verbose=1)
test_features = model.predict(test_generator, steps=len(test_generator), verbose=1)

train_labels=train_generator.classes
val_labels=validation_generator.classes
test_label=test_generator.classes

print("train_labels shape:", np.array(train_labels).shape)
print("val_labels shape:", np.array(test_label).shape)

trainval_features = np.concatenate(train_features, axis=0)
trainval_labels = np.concatenate((train_labels, test_label), axis =0)

for i, arr in enumerate(train_features[:3]):
    print(f"train_features[{i}] shape: {arr.shape}")

print("Number of train feature arrays:", len(train_features))

for i, arr in enumerate(trainval_features[:3]):
    print(f"val_features[{i}] shape: {arr.shape}")

print(type(trainval_features))  # should be list
print(type(trainval_features[0]))  # likely scalar now (int/float)
print(trainval_features[:20])  # print first 20 elements

print("Number of val feature arrays:", len(trainval_features))
val_features_array = np.array(train_features).reshape(-1, 1)
val_features = [arr for arr in val_features_array]
len(val_features) == 1279
val_features[0].shape == (4,)
# Convert train_features to 2D array (10240, 4)
train_features_array = np.array(train_features)  # shape: (10240, 4)

# Convert flat val_features to 2D array (1279, 4)
val_features_array = np.array(val_features[:1279 * 4]).reshape(1279, 4)

# Combine features

# From your working train data
train_features_array = np.array(train_features)  # (10240, 4)
train_labels_array = np.array(train_labels)

# Reshape val features safely
val_sample_count = len(val_labels)  
val_features_array = np.vstack(val_features)
val_labels_array = np.array(val_labels)

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D


base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")


x = base_model.output                
x = GlobalAveragePooling2D()(x)      
feature_extractor = Model(inputs=base_model.input, outputs=x)


train_generator = train_datagen.flow_from_directory(
    directory=r"C:\Kaggle\Input\augmented-alzheimer-mri-dataset-v2\Alzheimer_Dataset\test",
    target_size=(224, 224),   
    batch_size=32,
    class_mode="categorical",
    shuffle=False            
)

features = feature_extractor.predict(train_generator, verbose=1)
labels = train_generator.classes


base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=x)

train_features_array = feature_extractor.predict(train_generator, verbose=1)
val_features_array   = feature_extractor.predict(validation_generator, verbose=1)

def aggregate_features(features_array, labels, n_features_per_image=None):
    import os
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D

    DATA_DIR = r"C:\Kaggle\Input\augmented-alzheimer-mri-dataset-v2\Alzheimer_Dataset"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    datagen = ImageDataGenerator(rescale=1. / 255)

    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D()(base_model.output)  #  one vector per image
    feature_extractor = Model(inputs=base_model.input, outputs=x)

    train_features = feature_extractor.predict(train_generator, verbose=1)
    val_features = feature_extractor.predict(validation_generator, verbose=1)

    train_labels = train_generator.classes
    val_labels = val_generator.classes

    print("Train features:", train_features.shape)
    print("Train labels:", train_labels.shape)
    print("Val features:", val_features.shape)
    print("Val labels:", val_labels.shape)

    trainval_features = np.concatenate([train_features, val_features], axis=0)
    trainval_labels = np.concatenate([train_labels, val_labels], axis=0)

    print("Final features:", trainval_features.reshape(-1, trainval_labels))
    print("Final labels:", trainval_labels.shape)

print("Done test X")
print("Final features:", trainval_features.shape)
print("Final labels:", trainval_labels.shape)

print("Train features:", train_features_array.shape)
print("Val features:", val_features_array.shape)

trainval_features = np.concatenate([train_features_array, val_features_array], axis=0)
trainval_labels   = np.concatenate([train_labels, val_labels], axis=0)

print("Combined features:", trainval_features.shape)
print("Combined labels:", trainval_labels.shape)


trainval_features = np.concatenate([train_features_array, val_features_array], axis=0)
trainval_labels   = np.concatenate([train_labels, val_labels], axis=0)

print("trainval_features:", trainval_features.shape)
print("trainval_labels:", trainval_labels.shape)

import math

steps_train = math.ceil(train_generator.samples / train_generator.batch_size)
steps_val   = math.ceil(validation_generator.samples / validation_generator.batch_size)

train_features_array = feature_extractor.predict(train_generator, steps=steps_train, verbose=1)
val_features_array   = feature_extractor.predict(validation_generator, steps=steps_val, verbose=1)

train_labels = train_generator.classes
val_labels   = validation_generator.classes

trainval_features = np.concatenate([train_features_array, val_features_array], axis=0)
trainval_labels   = np.concatenate([train_labels, val_labels], axis=0)


print("trainval_features shape:", trainval_features.shape)
print("trainval_labels shape:", trainval_labels.shape)
assert trainval_features.shape[0] == trainval_labels.shape[0], "Still mismatched!"

print("All good! Ready for training.")

X_train_2d = trainval_features.reshape(trainval_features.shape[0], -1)
X_test_2d = test_features.reshape(test_features.shape[0], -1)
indices = np.random.permutation(X_train_2d.shape[0])
shuffled_X_train = X_train_2d[indices]
shuffled_X_labels = trainval_labels[indices]

import pandas as pd
print("Final training label distribution:")


print("Done test 4")

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01]
}

print("Done Test 5")

xgb_model = xgb.XGBClassifier()
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3)
grid_search.fit(shuffled_X_train, shuffled_X_labels)


best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Done Test 6")

print(best_params)
print(best_score)



best_xgb = xgb.XGBClassifier(**best_params)
best_xgb.fit(shuffled_X_train, shuffled_X_labels)


X_test = feature_extractor.predict(test_generator)

print("X_test shape:", X_test.shape)  

# Predict with XGBoost
y_pred = best_xgb.predict(X_test)

print("X_train_2d shape:", X_train_2d.shape)
print("trainval_labels shape:", trainval_labels.shape)

import pandas as pd
from sklearn.metrics import classification_report

# Convert to Series
test_label_series = pd.Series(test_label).dropna()
y_pred_series = pd.Series(y_pred).dropna()



# Trim to match length
min_len = min(len(test_label_series), len(y_pred_series))
test_label_series = test_label_series.iloc[:min_len]
y_pred_series = y_pred_series.iloc[:min_len]

# Report
print(pd.Series(test_label_series).value_counts())
print(pd.Series(y_pred_series).value_counts())



# Generate classification report

report = classification_report(test_label_series, y_pred_series)
print("Classification Report:")
print(report)
# Compute the classification report

from itertools import product

# Compute the confusion matrix
confusion_mtx = confusion_matrix(test_label_series, y_pred_series)

print("Confusion Matrix:")
print(confusion_mtx)

class_names = ['Mild', 'Moderate', 'Non', 'Very']

plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
fmt = 'd'
thresh = confusion_mtx.max() / 2.
for i, j in product(range(confusion_mtx.shape[0]), range(confusion_mtx.shape[1])):
    plt.text(j, i, format(confusion_mtx[i, j], fmt),
             horizontalalignment="center",
             color="white" if confusion_mtx[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
