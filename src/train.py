import os
import cv2
import pickle
import argparse
import numpy as np
from tensorflow import keras
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="src/dataset")
    parser.add_argument('--models', type=str, default="src/models") 
    parser.add_argument('--output_model', type=str, default="src/models")


    return parser.parse_args()

def encoder(y, num_classes):
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, num_classes)

    return y

def load_dataset(path, num_classes):
    x = []
    y = []

    image_paths = list(paths.list_images(path))

    for file in image_paths:
        file_name = file.split('/')[-1]
        file_name_without_ext = file_name.split('.')[0]
        file_name_without_ext = file_name.split('_')[0]
        image = cv2.imread(file)
        image = cv2.resize(image, (100, 100))
        x.append(image)
        y.append(file_name_without_ext)
    x = np.array(x, dtype="float") / 255.0
    y = encoder(np.array(y), num_classes) 
    return x, y

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def build_model(width, height, depth, classes):
    alpha = 0.3
    
    # Inputs
    inputs = keras.Input(shape = (width, height, depth))

    # Layer follows previous conv layer
    layer1_3x3 = keras.layers.Convolution2D(32,  (3, 3), padding="valid", activation=None, strides=(2,2))(inputs)
    layer1_3x3_activation = keras.layers.BatchNormalization()(layer1_3x3) # keras.layers.LeakyReLU(alpha)(layer1_3x3)

    layer2_3x3 = keras.layers.Convolution2D(32,  (3, 3), padding="valid", activation=None, strides=(1,1))(layer1_3x3_activation)
    layer2_3x3_activation =  keras.layers.BatchNormalization()(layer2_3x3) #keras.layers.LeakyReLU(alpha)(layer2_3x3)

    layer3_3x3 = keras.layers.Convolution2D(64,  (3, 3), padding="same", activation=None, strides=(1,1))(layer2_3x3_activation)
    layer3_3x3_activation = keras.layers.BatchNormalization()(layer3_3x3)#keras.layers.LeakyReLU(alpha)(layer3_3x3)

    layer4_pool = keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding="valid")(layer3_3x3_activation)
    layer4_pool_activation = keras.layers.BatchNormalization()(layer4_pool)

    layer4_3x3 = keras.layers.Convolution2D(96,  (3, 3), padding="valid", activation=None, strides=(2,2))(layer3_3x3_activation)
    layer4_3x3_activation = keras.layers.BatchNormalization()(layer4_3x3)#keras.layers.LeakyReLU(alpha)(layer4_3x3)

    layer_concatening_1 = keras.layers.Concatenate()([layer4_pool_activation, layer4_3x3_activation])

    # left branch 
    layer6_1x1_l = keras.layers.Convolution2D(64,  (1, 1), padding="same", activation=None, strides=(1,1))(layer_concatening_1)
    layer6_1x1_l_activation = keras.layers.BatchNormalization()(layer6_1x1_l)

    layer7_3x3_l = keras.layers.Convolution2D(96,  (3, 3), padding="valid", activation=None, strides=(1,1))(layer6_1x1_l_activation)
    layer7_3x3_l_activation =  keras.layers.BatchNormalization()(layer7_3x3_l) #keras.layers.LeakyReLU(alpha)(layer7_3x3_l)

    # right bramch
    layer6_1x1_r = keras.layers.Convolution2D(64,  (1, 1), padding="same", activation=None, strides=(1,1))(layer_concatening_1)
    layer6_1x1_r_activation = keras.layers.BatchNormalization()(layer6_1x1_r)

    layer7_7x1_r = keras.layers.Convolution2D(64,  (7, 1), padding="same", activation=None, strides=(1,1))(layer6_1x1_r_activation)
    layer8_1x7_r = keras.layers.Convolution2D(64, (1, 7), padding="same", activation=None, strides=(1,1))(layer7_7x1_r)

    layer8_1x7_r_activation = keras.layers.BatchNormalization()(layer8_1x7_r)

    layer9_3x3_r = keras.layers.Convolution2D(96, (3, 3), padding="valid", activation=None, strides=(1,1))(layer8_1x7_r_activation)
    layer9_3x3_r_activation = keras.layers.BatchNormalization()(layer9_3x3_r)#keras.layers.LeakyReLU(alpha)(layer9_3x3_r)

    layer_concatening_2 = keras.layers.Concatenate()([layer7_3x3_l_activation, layer9_3x3_r_activation])

    # Flatten
    flat = keras.layers.GlobalMaxPooling2D()(layer_concatening_2)

    # Outputs
    outs = []
    outs.append(keras.layers.Dense(classes, activation='softmax')(flat))

    # Final model definition
    model = Model(inputs=inputs, outputs=outs)
    return model


def main():
    learning_rate = 1e-3
    batch_size = 64
    epochs = 100
    args = parse_args()
    train_dataset = os.path.join(args.dataset, 'training')
    test_dataset = os.path.join(args.dataset, 'testing')
    validation_dataset = os.path.join(args.dataset, 'validation')
    encoder_path = os.path.join(args.models, 'encoder.pkl')

    with open(encoder_path, 'rb') as f:
        loaded_label_encoder = pickle.load(f)
    
    classes = loaded_label_encoder.classes_
    num_classes = len(classes)

    train_x, train_y = load_dataset(train_dataset, num_classes)
    test_x, test_y = load_dataset(test_dataset, num_classes)
    validation_x, validation_y = load_dataset(validation_dataset, num_classes)

    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

    model = build_model(100, 100, 3, num_classes)
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=lr_schedule(0)),
	metrics=["accuracy"])

    print("[INFO] training network for {} epochs...".format(epochs))
    lr_scheduler = LearningRateScheduler(lr_schedule)
    H = model.fit(aug.flow(x=train_x, y=train_y, batch_size=batch_size),
              validation_data=(validation_x, validation_y),
              epochs=epochs)
    
    print("[INFO] evaluating network...")
    predictions = model.predict(x=test_x)
    count = 0
    for  i in range(len(predictions)):
        if test_y.argmax(axis=1)[i] == predictions.argmax(axis=1)[i]:
            count += 1
    
    acuracia = count/len(test_y)
    with open(os.path.join(args.output_model, 'output.txt'), 'w') as f:
        # Escreve a acurácia
        f.write(f"Acurácia: {acuracia * 100:.2f} %\n")
        f.write('_ ' * len('Esperado\tPredito') + '\n')      
        f.write("Esperado\tPredito\n")
        for esperado, predito in zip(test_y.argmax(axis=1), predictions.argmax(axis=1)):
            f.write(f"{classes[esperado]}\t{classes[predito]}\n")
    
    # save the network to disk
    print("[INFO] serializing network to '{}'...".format(args.output_model))
    model.save(os.path.join(args.output_model, 'face_recognizer.h5'))
    
if __name__ == "__main__":
    main()
