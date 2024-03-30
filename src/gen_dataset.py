import os
import cv2
import pickle
import shutil
import imutils
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imutils import paths

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_images_processed', type=str, default="src/images_processed")
    parser.add_argument('--dataset', type=str, default="src/dataset")
    parser.add_argument('--models', type=str, default="src/models")
    parser.add_argument('--train-pct', type=float, default=0.8)
    parser.add_argument('--test-pct', type=float, default=0.1)  

    return parser.parse_args()

def create_labels(image_paths):
    names = []
    for path in image_paths:
        name = path.split("/")[-2]
        if name not in names:
            names.append(name)

    return names

def create_dataset(image_path, path_to_save):
    file_name = image_path.split('/')[-1]
    file_name = os.path.join(path_to_save, file_name)
    image = cv2.imread(image_path)
    cv2.imwrite(file_name, image)


def main():
    args = parse_args()
    le = LabelEncoder()
    train_pct = args.train_pct
    test_pct = args.test_pct
    processed_image_paths = args.path_images_processed
    dataset = args.dataset
    path_to_save_models = args.models
    train_path = os.path.join(dataset,'training')
    test_path = os.path.join(dataset,'testing')
    validation_path = os.path.join(dataset,'validation')

    if os.path.exists(dataset):
        shutil.rmtree(dataset)
    
    print('Creating directories for datasets...')
    os.mkdir(dataset)
    os.makedirs(train_path)
    os.makedirs(test_path)
    os.makedirs(validation_path)

    if os.path.exists(path_to_save_models):
        shutil.rmtree(path_to_save_models)    
    os.makedirs(path_to_save_models)

    image_paths = list(paths.list_images(processed_image_paths))
    np.random.shuffle(image_paths)
    total_images = len(image_paths)

    print()
    print('spliting dataset...')

    count = 0
    for path in image_paths:
        if len(os.listdir(train_path)) < total_images * train_pct:
            create_dataset(path, train_path)
        elif len(os.listdir(test_path)) < total_images * test_pct:
            create_dataset(path, test_path)
        else:
            create_dataset(path,validation_path)
        count += 1
        print(f'Total number of processed images: {count}', end='\r')

    print('\n')
    print(f'Training dataset size: {len(os.listdir(train_path))}')
    print(f'Testing dataset size: {len(os.listdir(test_path))}')
    print(f'Validation dataset size: {len(os.listdir(validation_path))}')
    print()

    print('creating labels...')
    labels = create_labels(image_paths)
    labels = le.fit_transform(labels)

    print("Saving label encoding ...")

    f = open(os.path.join(path_to_save_models, 'encoding'), "wb")
    f.write(pickle.dumps(le))
    f.close()    

if __name__ == "__main__":
    main()