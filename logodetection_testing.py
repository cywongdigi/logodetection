import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Define the target size
TARGET_SIZE = 224

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def load_test_images(directory):
    images = []
    lbp_features = []
    labels = []
    file_paths = []
    label_dict = {}

    for subdir, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(subdir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Failed to read image {image_path}")
                    continue
                image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
                normalized_image = normalize_image(image)
                images.append(normalized_image)
                lbp = extract_lbp_features(image)
                lbp_features.append(lbp)
                label = os.path.basename(subdir)
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
                labels.append(label_dict[label])
                file_paths.append(image_path)

    return np.array(images), np.array(lbp_features), np.array(labels), label_dict

def extract_resnet_features(images, resnet, device):
    resnet.eval()
    resnet = resnet.to(device)
    features = []
    for image in images:
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device).float()
        with torch.no_grad():
            feature = resnet(image).cpu().numpy().flatten()
        features.append(feature)
    return np.array(features)

def calculate_specificity(cm):
    tn = np.sum(np.diag(cm)) - np.sum(cm, axis=1)
    fp = np.sum(cm, axis=0) - np.diag(cm)
    specificity = np.divide(tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    return specificity.mean()

class LogoClassifierEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogoClassifierEnsemble, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_dim)

        self.fc3 = nn.Linear(input_dim, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)

        x2 = self.fc3(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc4(x2)

        x = (x1 + x2) / 2
        return x

def evaluate_model(test_images_dir, model_path, label_dict_path, scaler_path):
    print("[INFO] Loading the test dataset...")
    images, lbp_features, labels, label_dict = load_test_images(test_images_dir)
    print("[INFO] Finished loading the test dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Extracting ResNet-50 features...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet_features = extract_resnet_features(images, resnet, device)
    print("[INFO] Finished extracting ResNet-50 features.")

    print("[INFO] Combining features...")
    combined_features = np.hstack((resnet_features, lbp_features))

    scaler = joblib.load(scaler_path)
    combined_features = scaler.transform(combined_features)
    combined_features = combined_features.astype(np.float32)
    labels = labels.astype(np.int64)
    print("[INFO] Finished combining and scaling features.")

    print("[INFO] Loading the trained model...")
    model = LogoClassifierEnsemble(input_dim=combined_features.shape[1], output_dim=len(label_dict))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()
    print("[INFO] Finished loading the trained model.")

    with torch.no_grad():
        inputs = torch.tensor(combined_features).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)
    specificity = calculate_specificity(cm)

    print(f"\n[INFO] Evaluation Metrics")
    print(f"[INFO] ==================")
    print(f"[INFO] Accuracy: {accuracy}")
    print(f"[INFO] Precision: {precision}")
    print(f"[INFO] Recall: {recall}")
    print(f"[INFO] Specificity: {specificity}")
    print(f"[INFO] F1 Score: {f1}")
    print(f"[INFO] Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    test_images_dir = 'dataset/test'
    model_path = 'logodetection.pth'
    label_dict_path = 'label_dict.pkl'
    scaler_path = 'scaler.pkl'

    evaluate_model(test_images_dir, model_path, label_dict_path, scaler_path)
