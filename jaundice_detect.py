import cv2
import pickle
from utils import prediction_pipeline
from ultralytics import YOLO

with open('model/jaundice_predicter.pkl', 'rb') as f:
    jaundice_predicter = pickle.load(f)

# Load the YOLO model
sclera_detector = YOLO('model/sclera_detector.pt')

image_path = 'images/test_img.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image , sclera_image , yellow_mask , JI , label = prediction_pipeline(
    image, 
    sclera_detector, 
    jaundice_predicter
)

print(label)
