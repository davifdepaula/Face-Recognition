import os
import cv2
import pickle
import tensorflow
import argparse
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="src/dataset")
    parser.add_argument('--models', type=str, default="src/models") 
    parser.add_argument('--output_model', type=str, default="src/models")


    return parser.parse_args()

def load_dataset(path):
    x = []
    y = []
    image_paths = list(paths.list_images(path))
    for file in image_paths:
        file_name = file.split('/')[-1]
        file_name_without_ext = file_name.split('.')[0]
        file_name_without_ext = file_name.split('_')[0]
        image = cv2.imread(file)
        image = cv2.resize(image, (64, 64))
        x.append(image)
        y.append(file_name_without_ext)
    x = np.array(x, dtype="float") / 255.0
    return x, y


def build_model(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    
    model.add(Conv2D(16, (3, 3), padding="same",
			input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    # return the constructed network architecture
    return model


def main():
    learning_rate = 1e-4
    batch_size = 64
    epochs = 100

    args = parse_args()
    le = LabelEncoder()
    
    train_dataset = os.path.join(args.dataset, 'training')
    test_dataset = os.path.join(args.dataset, 'testing')
    validation_dataset = os.path.join(args.dataset, 'validation')

    train_x, train_y = load_dataset(train_dataset)
    test_x, test_y = load_dataset(test_dataset)
    validation_x, validation_y = load_dataset(validation_dataset)

    encoder_path = os.path.join(args.models, 'encoder.pkl')

    with open(encoder_path, 'rb') as f:
        loaded_label_encoder = pickle.load(f)

    le = LabelEncoder()
    num_classes = len(loaded_label_encoder.classes_)

    train_y, test_y, validation_y = le.fit_transform(train_y), le.fit_transform(test_y), le.fit_transform(validation_y)
    train_y, test_y, validation_y = to_categorical(train_y, num_classes), to_categorical(test_y, num_classes ), to_categorical(validation_y, num_classes)
    
    # opt = Adam(lr=learning_rate)
    model = build_model(64, 64, 3, num_classes)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate),
	metrics=["accuracy"])

    print("[INFO] training network for {} epochs...".format(epochs))
    H = model.fit(x=train_x, y=train_y, batch_size=batch_size,
              validation_data=(validation_x, validation_y),
              steps_per_epoch= len(train_x) // epochs, epochs=epochs)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(x=test_x)
    acuracia = round(accuracy_score(test_y.argmax(axis=1),
        predictions.argmax(axis=1)), 2) * 100
    print(f'acuracia do modelo: {acuracia}')
    
    # save the network to disk
    print("[INFO] serializing network to '{}'...".format(args.output_model))
    model.save(os.path.join(args.output_model, 'face_recognizer.h5'))
    
if __name__ == "__main__":
    main()
