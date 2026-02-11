import cv2
import os

def capture_photo(nom_fichier):
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        if not os.path.exists('photos'):
            os.makedirs('photos')
        path = f'photos/{nom_fichier}.jpg'
        cv2.imwrite(path, frame)
        cam.release()
        cv2.destroyAllWindows()
        return path
    cam.release()
    cv2.destroyAllWindows()
    return None
