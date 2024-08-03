import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.feature import local_binary_pattern
import joblib
from collections import defaultdict

# Define the target size
TARGET_SIZE = 224

# Function to normalize image
def normalize_image(image):
    return image.astype(np.float32) / 255.0

# Function to preprocess and load the model
def load_model(model_path, in_features, num_classes):
    classifier = nn.Linear(in_features, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(device)
    classifier.eval()
    return classifier, device

# Function to extract ResNet-50 features
def extract_resnet_features(image, resnet, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = resnet(image).cpu().numpy().flatten()
    return feature

# Function to extract LBP features
def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = (gray_image * 255).astype(np.uint8)  # Ensure image is of integer dtype
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# Function to predict logo
def predict_logo(classifier, resnet, scaler, device, label_dict, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to read image {image_path}")
        return None
    image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
    image = normalize_image(image)

    resnet_feature = extract_resnet_features(image, resnet, device)
    lbp_feature = extract_lbp_features(image)
    combined_features = np.hstack((resnet_feature, lbp_feature))
    combined_features = scaler.transform([combined_features])

    inputs = torch.tensor(combined_features, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = classifier(inputs)
        _, predicted_class = torch.max(outputs, 1)

    for label, index in label_dict.items():
        if index == predicted_class.item():
            return label
    return None

# Function to predict logos in all images in a folder and calculate accuracy
def predict_logos_in_folder(classifier, resnet, scaler, device, label_dict, folder_path):
    brand_accuracy = defaultdict(lambda: {"total": 0, "correct": 0})
    overall_total = 0
    overall_correct = 0

    for subdir, _, files in os.walk(folder_path):
        folder_name = os.path.basename(subdir)
        if folder_name not in label_dict:
            continue  # Skip folders not in label_dict
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, filename)
                predicted_label = predict_logo(classifier, resnet, scaler, device, label_dict, image_path)
                print(f"Image: {filename}, Predicted logo: {predicted_label}, Actual folder: {folder_name}")
                brand_accuracy[folder_name]["total"] += 1
                overall_total += 1
                if predicted_label == folder_name:
                    brand_accuracy[folder_name]["correct"] += 1
                    overall_correct += 1

    for brand, data in brand_accuracy.items():
        accuracy = (data["correct"] / data["total"]) * 100 if data["total"] > 0 else 0
        print(f"Brand: {brand}, Total images: {data['total']}, Correct predictions: {data['correct']}, Accuracy: {accuracy:.2f}%")

    overall_accuracy = (overall_correct / overall_total) * 100 if overall_total > 0 else 0
    print(f"\nOverall total images: {overall_total}, Overall correct predictions: {overall_correct}, Overall accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    model_path = 'classifier_logo_detection.pth'  # Path to your saved model
    label_dict_path = 'label_dict.pkl'  # Path to your saved label dictionary
    scaler_path = 'scaler.pkl'  # Path to your saved scaler
    folder_path = 'dataset/test'  # Path to the folder containing images to predict

    # Load the label dictionary and scaler
    label_dict = joblib.load(label_dict_path)
    scaler = joblib.load(scaler_path)

    # Determine the number of input features for the classifier
    dummy_image_path = None
    for subdir, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                dummy_image_path = os.path.join(subdir, filename)
                break
        if dummy_image_path:
            break

    if not dummy_image_path:
        raise FileNotFoundError("No images found in the provided folder path.")

    dummy_image = cv2.imread(dummy_image_path)
    dummy_image = cv2.resize(dummy_image, (TARGET_SIZE, TARGET_SIZE))
    dummy_image = normalize_image(dummy_image)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
    resnet = resnet.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)
    resnet_feature = extract_resnet_features(dummy_image, resnet, device)
    lbp_feature = extract_lbp_features(dummy_image)
    combined_features = np.hstack((resnet_feature, lbp_feature))
    in_features = combined_features.shape[0]

    # Load the classifier model and device
    classifier, device = load_model(model_path, in_features, len(label_dict))

    # Load the ResNet-50 model for feature extraction
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
    resnet = resnet.to(device)
    resnet.eval()

    # Predict logos in all images in the specified folder and calculate accuracy
    predict_logos_in_folder(classifier, resnet, scaler, device, label_dict, folder_path)
