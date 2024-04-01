import cv2
import numpy as np
from mtcnn import MTCNN

class Face_detector:
     
    def get_face_postion(self, image):
        face_classifier = MTCNN()

        face_map = face_classifier.detect_faces(image)
        if len(face_map) > 0:
            face_positions = face_map[0]['box']
            return face_positions
            
    def crop_face(self, image, face_positions):
        x, y, w, h = face_positions
        cropped_image = image[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (250, 250))
        
        return cropped_image
    
    def save_cropped_face(self, image, path_to_save):
        cv2.imwrite(path_to_save, image)
