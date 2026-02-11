# -*- coding: utf-8 -*-

#pip install torch torchvision torchaudio
#pip install imutils
#pip install websockets
#import torch
#print(torch.__version__)  # Verify PyTorch version

#pip install ipython numpy opencv-python torch Pillow scipy websockets imutils
#pip install dlib-19.24.99-cp312-cp312-win_amd64.whl

#pip install Flask

from IPython.display import display, Javascript, Image
from base64 import b64encode
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import PIL
from PIL import Image as PILImage
import sys
import io
import os
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import json
import sys
import asyncio
import websockets
import base64

captureCount = 0  # Global counter to keep track of the number of images captured

# The main code
#print("Python script started.")  # Debug statement
#sys.stdout.flush()  # Ensure output is flushed immediately

# Add the path to the directory containing MainModel.py
model_dir = r'C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\modelDownload'
sys.path.append(model_dir)

# Import the MainModel class
#try:
from MainModel import KitModel
    #print("MainModel imported successfully.")
#except ModuleNotFoundError as e:
    
    #print(f"Error importing MainModel: {e}")


# Set the class label
id2class = {1: 'Detected', 0: 'Please remove the mask'}


# Function to generate the anchors
def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes

# Function to decode the model output
def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox

# Function to remove the overlap of anchor
def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    if len(bboxes) == 0:
        return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]
    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        if keep_top_k != -1 and len(pick) >= keep_top_k:
            break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]


# Paths to the model files on your computer
local_paths = {
    'shape_predictor_68_face_landmarks.dat': r'C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\modelDownload\shape_predictor_68_face_landmarks.dat',
}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(local_paths['shape_predictor_68_face_landmarks.dat'])
face_detector = dlib.get_frontal_face_detector()

device = "cuda" if torch.cuda.is_available() else 'cpu'
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)

# Load the model
model = torch.load(r"C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\modelDownload\model360.pth", map_location=device)
model = model.to(device)

def enhance_image(image, tier='none'):
    
    #print(f"Enhancement tier received: {tier}")
    def least_enhancement(img):
    # Least enhancement: Simple histogram equalization for contrast adjustment
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return img

    def basic_enhancement(img):
        # Basic enhancement: Apply Gaussian Blur and Unsharp Masking (Sharpening)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        sharpened = cv2.addWeighted(img, 1.3, blurred, -0.3, 0)
        
        # Optional: Slight contrast enhancement
        img = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
        return img
    
    
    def upgraded_enhancement(img):
        # Upgraded enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        img = cv2.inpaint(img, np.zeros(img.shape[:2], dtype=np.uint8), 3, cv2.INPAINT_TELEA)
        return img

    def advanced_enhancement(img):
        # Advanced enhancement: Combine CLAHE and additional scaling
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        img = cv2.inpaint(img, np.zeros(img.shape[:2], dtype=np.uint8), 3, cv2.INPAINT_TELEA)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=-20)
        return img

    # Convert tier to string and ensure it's in lowercase
    tier = str(tier).lower()

    if tier == 'none':
        return least_enhancement(image)
    elif tier == 'basic':
        return basic_enhancement(image)
    elif tier == 'upgraded':
        return upgraded_enhancement(image)
    elif tier == 'advanced':
        return advanced_enhancement(image)
    else:
        raise ValueError("Invalid tier. Choose from 'none', 'basic', 'upgraded', or 'advanced'.")
    
# Function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGB')
    iobuf = io.BytesIO()
    bbox_PIL.save(iobuf, format='png')
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))
    return bbox_bytes

# Function to check if the image is blurry
def is_blurry(img, threshold=0.3):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy array, but got {type(img)}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blurry = laplacian_var < threshold
    return is_blurry, laplacian_var

# Function to check if the image is too dark
def is_too_dark(img, threshold=0.3):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy array, but got {type(img)}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    is_dark = mean_brightness < threshold
    return is_dark, mean_brightness

# Function to check if the image is too bright
def is_too_bright(img, threshold=200):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected a NumPy array, but got {type(img)}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    is_bright = mean_brightness > threshold
    return is_bright, mean_brightness


def is_facing_forward(image, threshold=0.1):
    face_features = extract_face_features(image)

    if not face_features:
        return False, "No face detected"

    features = face_features[0]

    left_eye = features[36]
    right_eye = features[45]
    nose = features[30]

    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_to_left_eye_distance = np.linalg.norm(nose - left_eye)
    nose_to_right_eye_distance = np.linalg.norm(nose - right_eye)

    eye_nose_ratio = abs(nose_to_left_eye_distance - nose_to_right_eye_distance) / eye_distance

    is_facing = eye_nose_ratio < threshold
    return is_facing, "Please face forward and look directly at the camera."

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Integrating GAN
import os
import zipfile
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms

# Set logging level
import logging

# Set logging level
logging.basicConfig(level=logging.ERROR)

# Load the discriminator 
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def load_model(gan_model, path, gan_device):
    gan_model.load_state_dict(torch.load(path, map_location=gan_device))
    gan_model.to(gan_device)
    return gan_model

# Parameters used in the gan_model
image_size = 64
ngpu = 1
nz = 100
ndf = 64
nc = 3

# line 156 has another device, check if its the same as this (its diff)
gan_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gan_model_path = r"C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\modelDownload\disc_stop_15_lr1.pth"

# Check if the file exists
if not os.path.isfile(gan_model_path):
    print(f"Error: The file at {gan_model_path} does not exist.")
else:
    print(f"File exists: {gan_model_path}")

gan_model = Discriminator(ngpu).to(gan_device)


gan_model = load_model(gan_model, gan_model_path, gan_device)
gan_model.eval()

def preprocess_image_GAN(image_path, image_size):
    # Open and convert the image to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# put in 661
# Function to predict real or fake using the GAN model
def is_fake_or_real(gan_model, image_path, gan_device, image_size):
    # Preprocess the image
    image_tensor = preprocess_image_GAN(image_path, image_size).to(gan_device)
    
    # Predict
    with torch.no_grad():
        output = gan_model(image_tensor).view(-1).item()
        
    threshold = 0.3
    
    #return output > threshold # Return True if the image is real, False otherwise (somehow this returns real image as fake so)
    return output < threshold #for testing purposes (apparently this is the correct one..)

def js_to_image(js_reply):
    if isinstance(js_reply, dict) and 'img' in js_reply:
        # Extract the NumPy array from the dictionary
        return np.array(js_reply['img'])
    elif isinstance(js_reply, np.ndarray):
        # Directly return the NumPy array
        return js_reply
    else:
        raise ValueError(f"Expected a dictionary with a NumPy array under the key 'img' or a NumPy array directly, but got {type(js_reply)}.")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to extract face features
def extract_face_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    features = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)
        features.append(shape_np)
    return features

from torchvision import transforms  # Add this import statement
# Pre-trained mask detection model
def detect_mask(image, model, device, confidence_threshold=0.5):
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    
    # Perform prediction
    with torch.no_grad():
        output = model(image)
        # If the model returns a tuple, extract the relevant part
        if isinstance(output, tuple):
            output = output[0]
        
        # Print the shape of the output for debugging
        #print(f'Output shape: {output.shape}')
        
        # Assuming output has shape [1, num_detections, 4] where 4 represents (x, y, width, height) or (x, y, score, class)
        # Extract scores and class predictions
        scores = output[0, :, 2]  # Example for the score, adjust index based on your model's output
        class_predictions = output[0, :, 3]  # Example for the class predictions
        
        # Filter out low confidence detections
        mask_indices = torch.where((scores > confidence_threshold) & (class_predictions == 1))[0]  # Assuming class '1' is 'Mask'
        
        # If there are any mask detections
        if len(mask_indices) > 0:
            return 'Please remove your mask'
        else:
            return 'No Mask'



# Function to check if the person is blinking
def is_blinking(image):
    try:
        face_features = extract_face_features(image)
        #if len(face_features) == 0:
         #   return False, "No face detected."
        for features in face_features:
            left_eye = features[36:42]
            right_eye = features[42:48]
            left_eye_aspect_ratio = (dist.euclidean(left_eye[1], left_eye[5]) + dist.euclidean(left_eye[2], left_eye[4])) / (2.0 * dist.euclidean(left_eye[0], left_eye[3]))
            right_eye_aspect_ratio = (dist.euclidean(right_eye[1], right_eye[5]) + dist.euclidean(right_eye[2], right_eye[4])) / (2.0 * dist.euclidean(right_eye[0], right_eye[3]))
            if left_eye_aspect_ratio < 0.25 or right_eye_aspect_ratio < 0.25:
                return True, "Blink detected."
        return False, "Please blink your eyes."
    except Exception as e:
        return False, str(e)

# Import functions and variables from ImageCheckingSystem.py
from ImageCheckingSystem import calculate_average_feature_vector,extract_feature, preprocess_image, compare_features,read_excel_file, excel_file_path

# Directory containing the images
image_directory = r'C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\CapturedImages'


sheet_data = read_excel_file(excel_file_path)

import json
import numpy as np

def load_representative_features(filename='representative_features.json'):
    with open(filename, 'r') as f:
        features = json.load(f)
        # Convert lists back to numpy arrays
        return [(person, np.array(feature)) for person, feature in features]

# Load representative features before using them
#representative_features = load_representative_features(r"C:\xampp\htdocs\ImageCheckingSystem\Scripts\representative_features.json")

def extract_feature_vector(image_path):
    try:
        image = PILImage.open(image_path)
        processed_image = preprocess_image(image, image_path)
        if processed_image is None:
            return None, None  # Image was rejected in preprocessing

        feature_vector = extract_feature(processed_image)
        if feature_vector is None:
            print(f"No face feature vector found in {image_path}")
            return None, None

        print(f"Extracted feature vector for {os.path.basename(image_path)}: {feature_vector}")
        return feature_vector, processed_image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None




# Define the function to generate representative features for the two latest images
def generate_representative_feature(image_directory):
    # Get a list of all image files in the directory
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    # Sort files by last modified date (latest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Get the first and second latest images
    if len(image_files) < 2:
        print("Not enough images in the directory.")
        return None

    latest_image = image_files[0]
    second_latest_image = image_files[1]
    images = [latest_image, second_latest_image]
    
    features = []

    for img_path in images:
        img = PILImage.open(img_path)  # Use PIL Image here
        feature = extract_feature(img)
        if feature is not None:
            features.append(feature)
        else:
            print(f"No face feature vector found in {img_path}.")
            # Optionally delete the image if no feature is found

    if not features:
        print("No features extracted from any image.")
        return None

    features_array = np.array(features)

    # Compute average features
    average_features = np.mean(features_array, axis=0)
    return [("Webcam", average_features)]

async def process_images():
    threshold = 0.4
    # Load the representative features
    representative_features = load_representative_features(r"C:\xampp\htdocs\otp_test\FYP-Homies-FR-System\app\ImageCheckingSystem\Scripts\representative_features.json")
    if representative_features is None:
        return {'error': 'No representative features found.'}

    # Ensure representative_features is a list of tuples
    if not all(isinstance(item, tuple) and len(item) == 2 for item in representative_features):
        return {'error': 'Representative features are not in the expected format.'}

    # Get a list of all image files in the directory
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    # Sort files by last modified date (latest first)
    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Get the first and second latest images
    if len(image_files) > 0:
        latest_image = image_files[0]
        #second_latest_image = image_files[1]

        # Extract feature vectors and processed images
        latest_feature_vector, latest_processed_image = extract_feature_vector(latest_image)
        #second_latest_feature_vector, second_latest_processed_image = extract_feature_vector(second_latest_image)

        # Collect only approved feature vectors
        approved_feature_vectors = []
        if latest_feature_vector is not None:
            approved_feature_vectors.append(latest_feature_vector)
        #if second_latest_feature_vector is not None:
        #    approved_feature_vectors.append(second_latest_feature_vector)

        if not approved_feature_vectors:
            return {'error': 'No valid face features found. Please reset image count and try again.'}

        # Use the single approved feature vector as the average
        average_feature_vector = approved_feature_vectors[0]
        # Compare the average feature vector with the representative features
        best_match = None
        min_diff = float('inf')

        for person, rep_feature in representative_features:
            same_person, diff = compare_features(average_feature_vector, rep_feature, threshold=0.4)
            print(f"Comparing average feature vector with {person}: Difference = {diff}")
            if same_person and diff < min_diff:
                min_diff = diff
                best_match = person

        if best_match and min_diff < threshold:
            print(f"The best match is {best_match} with a difference of {min_diff}")
            print(f"Person is approved.")

            # Display the uploaded image with details from Google Sheets
            folder_name = best_match
            person_data = sheet_data[sheet_data['Name'] == folder_name]
            if not person_data.empty:
                person_data = person_data.iloc[0]
                name = person_data['Name']
                job_title = person_data['Job Title']
                department = person_data['Department']
                access_level = person_data['Access Level']
            else:
                name = folder_name
                job_title = 'Unknown'
                department = 'Unknown'
                access_level = 'Unknown'

            # Display the first captured image
            plt.imshow(latest_processed_image)
            plt.title(f"Name: {name}\nJob Title: {job_title}\nDepartment: {department}\nAccess Level: {access_level}\n")
            plt.axis('off')
            plt.show()

        else:
            print(f"No acceptable match found. The images are rejected.")

        return {
            'latest_image': latest_image,
            #'second_latest_image': second_latest_image
        }
    else:
        return {'error': 'Not enough images in the directory.'}

async def process_image(websocket, path, gan_model, gan_device, image_size):
    async def process_message(websocket, path):
        global captureCount  # Use the global counter

        async for message in websocket:
            data = json.loads(message)
            #print(f"Received message: {data}")  # Debugging log

            if 'captureCount' in data:
                captureCount = data['captureCount']
                print(f'Received captureCount: {captureCount}')
                await websocket.send(json.dumps({'status': 'captureCount received', 'captureCount': captureCount}))

            if data.get('type') == 'reset':
                captureCount = 0
                # Send success message and updated count
                response = json.dumps({'success': True, 'message': 'Image count reset successfully!', 'count': captureCount})
                await websocket.send(response)
                print(f'Capture count after reset: {captureCount}')
                continue

            image_data = data.get('image')
            tier = data.get('tier')

            # Decode the base64 image
            try:
                image_data = image_data.split(',')[1]
                frame = np.frombuffer(base64.b64decode(image_data), np.uint8)
                image = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            except Exception as e:
                continue

            if image is None:
                await websocket.send(json.dumps({'success': False, 'message': 'Invalid image data.'}))
                continue

            enhanced_image = enhance_image(image, tier)

            # Process the image
            blurry, blur_value = is_blurry(enhanced_image)
            if blurry:
                await websocket.send(json.dumps({'success': False, 'message': f'Please keep still, blurry frame detected. Blur value: {blur_value}'}))
                continue

            dark, brightness = is_too_dark(enhanced_image)
            if dark:
                await websocket.send(json.dumps({'success': False, 'message': f'Please ensure area is well-lit. Brightness: {brightness}'}))
                continue

            bright, brightness = is_too_bright(enhanced_image)
            if bright:
                await websocket.send(json.dumps({'success': False, 'message': f'Please turn down brightness. Brightness: {brightness}'}))
                continue

            facing_forward, facing_message = is_facing_forward(enhanced_image)
            if not facing_forward:
                await websocket.send(json.dumps({'success': False, 'message': facing_message}))
                continue

            # Convert frame to PIL image for mask detection
            pil_image = PILImage.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))

            # Detect mask
            mask_label = detect_mask(pil_image, model, device)

            # Send the result back to the frontend only if a mask is detected
            if mask_label == 'Mask':
                await websocket.send(json.dumps({'success': False, 'message': 'Please remove your mask.'}))
                continue

            blink_detected, blink_message = is_blinking(enhanced_image)
            if not blink_detected:
                await websocket.send(json.dumps({'success': False, 'message': 'Please blink to confirm you are a real person.'}))
                continue

            # Save the frame when face is detected and all conditions are met
            print(f"Before incrementing, captureCount: {captureCount}")
            if captureCount < 2:
                captureCount += 1
                print(f"After incrementing, captureCount: {captureCount}")
                file_path = f"C:/xampp/htdocs/otp_test/FYP-Homies-FR-System/app/ImageCheckingSystem/CapturedImages/captured_image{captureCount}.jpg"
                cv2.imwrite(file_path, enhanced_image)

                #--------------------------------            
                # Check if the image is fake or real
                is_real = is_fake_or_real(gan_model, file_path, gan_device, image_size)
                if not is_real:
                    await websocket.send(json.dumps({'success': False, 'message': 'Fake Person detected.'}))
                    captureCount-=1
                    continue
                #-------------------------------

                await websocket.send(json.dumps({'success': True, 'message': f'Face detected. Image is captured.'}))

                # Check if we have two images and send them to the frontend
                if captureCount == 1:
                    result = await process_images()
                    await websocket.send(json.dumps(result))

            else:
                #captureCount=1
                await websocket.send(json.dumps({'success': False, 'message': 'Image capture limit reached.'}))
                continue

    await process_message(websocket, path)
    

# Start the WebSocket server
start_server = websockets.serve(lambda websocket, path: process_image(websocket, path, gan_model, gan_device, image_size), "localhost", 9000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()