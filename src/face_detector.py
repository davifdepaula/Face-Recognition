import cv2
import numpy as np

class Face_detector:

    def __grayscale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def __distance(self, pointA, pointB):
        dist = np.linalg.norm(pointA - pointB)
        return dist
    
    def __get_min_distance(self, array):
        min = float("inf") 
        min_position = 0
        for i, position in enumerate(array):
            if self.__distance(position[1], position[2]) < min:
                min = self.__distance(position[1], position[2])
                min_position = position[:]
        return min_position
        
    def get_face_postion(self, image):
        gray_image = self.__grayscale_image(image)
        face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        face_positions = face_classifier.detectMultiScale(gray_image, 1.2, 5)
        if len(face_positions) == 0:
            return None
        
        if len(face_positions) == 1:
            return face_positions[0]
        
        if len(face_positions) > 1:
            return self.__get_min_distance(face_positions)
            
    def crop_face(self, image, face_positions):
        x, y, w, h = face_positions
        cropped_image = image[y:y+h, x:x+w]
        
        return cropped_image
    
    def save_cropped_face(self, image, path_to_save):
        cv2.imwrite(path_to_save, image)
