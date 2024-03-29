import cv2
from face_detector import Face_detector

path_to_image = "src/dataset_raw/Alexandra Daddario/Alexandra Daddario_0.jpg"
file_name = path_to_image.split('/')[-1]
image = cv2.imread(path_to_image)
detector = Face_detector()

face_positions = detector.get_face_postion(image)
cropped_face = detector.crop_face(image, face_positions)
detector.save_cropped_face(cropped_face, file_name)