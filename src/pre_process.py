import os
import cv2
from imutils import paths
import shutil
import argparse
from face_detector import Face_detector


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_images_raw', type=str, default="src/images_raw/")
    parser.add_argument('--path_images_processed', type=str, default="src/images_processed/")    

    return parser.parse_args()

def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def process_image(path_to_image):  
    image = cv2.imread(path_to_image)
    detector = Face_detector()
    face_positions = detector.get_face_postion(image)
    if face_positions is not None:
        cropped_face = detector.crop_face(image, face_positions)
        return cropped_face

def save_image(cropped_face, path_to_save):
    detector = Face_detector()
    detector.save_cropped_face(cropped_face, path_to_save)


def create_subdir(imagePaths, path_to_images, path_to_save):
    for sub_dir in imagePaths:
        subdir_path = "/".join(sub_dir.replace(path_to_images, path_to_save).split("/")[:3])
        if not os.path.exists(subdir_path):            
            os.makedirs(subdir_path)

def main():
    args = parse_args()
    
    path_to_images = args.path_images_raw
    path_to_save_processed_images  = args.path_images_processed

    if os.path.exists(path_to_save_processed_images):
        shutil.rmtree(path_to_save_processed_images)
        os.mkdir(path_to_save_processed_images)
    
    image_paths = list(paths.list_images(path_to_images))
    create_subdir(image_paths, path_to_images, path_to_save_processed_images)

    for sub_dir in image_paths:
        path_to_save = sub_dir.replace(path_to_images, path_to_save_processed_images)
        cropped_face = process_image(sub_dir)
        if cropped_face is not None:
            cropped_face = grayscale_image(cropped_face)
            save_image(cropped_face, path_to_save)

if __name__ == "__main__":
    main()