import cv2

class Face_detector:

    def __grayscale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_face_postion(self, image):
        gray_image = self.__grayscale_image(image)
        face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
        face_positions = face_classifier.detectMultiScale(gray_image, 1.3, 5)[0]

        return face_positions
    
    def crop_face(self, image, face_positions):
        x, y, h, w = face_positions
        cropped_image = image[y:y+h, x:x+w]
        
        return cropped_image
    
    def save_cropped_face(self, image, file_name, path_to_save = None):
        if path_to_save:
            cv2.imwrite(file_name, image, path_to_save)
            return
        cv2.imwrite(file_name, image)


    