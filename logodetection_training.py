import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from skimage.feature import local_binary_pattern
import joblib
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
import time

# Define the target size
TARGET_SIZE = 224

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def color_correction(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    corrected_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return corrected_image

def resize_and_pad_image(image, target_size=TARGET_SIZE):
    h, w, _ = image.shape
    if h > target_size or w > target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h))
    else:
        resized_image = image
        new_h, new_w = h, w

    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def smooth_image(image):
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed

def remove_and_recreate_dir(directory):
    if (os.path.exists(directory)):
        shutil.rmtree(directory)
    os.makedirs(directory)

def preprocess_images(input_dir, output_dir):
    for subdir, _, files in os.walk(input_dir):
        for filename in files:
            if (filename.endswith(('.jpg', '.jpeg', '.png'))):
                image_path = os.path.join(subdir, filename)
                image = cv2.imread(image_path)
                if (image is None):
                    print(f"[WARNING] Failed to read image {image_path}")
                    continue
                image = color_correction(image)
                image = resize_and_pad_image(image)
                image = sharpen_image(image)
                image = smooth_image(image)
                image = normalize_image(image)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if (not os.path.exists(output_subdir)):
                    os.makedirs(output_subdir)
                output_path = os.path.join(output_subdir, filename)
                cv2.imwrite(output_path, (image * 255).astype(np.uint8))

def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def load_images_with_lbp(directory):
    images = []
    lbp_features = []
    labels = []
    label_dict = {}
    file_paths = []  # New line to store file paths
    current_label = 0
    for subdir, _, files in os.walk(directory):
        for filename in files:
            if (filename.endswith(('.jpg', '.jpeg', '.png'))):
                image_path = os.path.join(subdir, filename)
                image = cv2.imread(image_path)
                if (image is None):
                    print(f"[WARNING] Failed to read image {image_path}")
                    continue
                image = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
                normalized_image = normalize_image(image)
                images.append(normalized_image)
                lbp = extract_lbp_features(image)
                lbp_features.append(lbp)
                label = os.path.basename(subdir)
                if (label not in label_dict):
                    label_dict[label] = current_label
                    current_label += 1
                labels.append(label_dict[label])
                file_paths.append(image_path)  # New line to append file path
    return np.array(images), np.array(lbp_features), np.array(labels), label_dict, file_paths  # Updated return statement

def calculate_specificity(cm):
    tn = np.sum(np.diag(cm)) - np.sum(cm, axis=1)
    fp = np.sum(cm, axis=0) - np.diag(cm)
    specificity = np.divide(tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    return specificity.mean()

# Additional data augmentation transforms
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

# Update CustomDataset class to use data augmentation transforms
class CustomDataset(Dataset):
    def __init__(self, images, lbp_features, labels, transform=None):
        self.images = images
        self.lbp_features = lbp_features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        lbp_feature = self.lbp_features[idx]
        if self.transform:
            image = self.transform(image)
        return image, lbp_feature, self.labels[idx]

# Define multiple models for ensemble
class LogoClassifierEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogoClassifierEnsemble, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_dim)

        # Additional models for ensemble
        self.fc3 = nn.Linear(input_dim, 256)
        self.fc4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)

        x2 = self.fc3(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.fc4(x2)

        # Combine outputs (simple average)
        x = (x1 + x2) / 2
        return x

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

def balance_classes(images, lbp_features, labels, file_paths, batch_size=100):
    smote = SMOTE(k_neighbors=2)
    combined_features = np.hstack((images.reshape(images.shape[0], -1), lbp_features))

    balanced_features = []
    balanced_labels = []
    balanced_file_paths = []  # New line to store file paths of balanced samples

    num_batches = int(np.ceil(combined_features.shape[0] / batch_size))

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, combined_features.shape[0])

        # Check if the batch contains more than one class
        if len(np.unique(labels[start:end])) > 1:
            try:
                batch_features, batch_labels = smote.fit_resample(combined_features[start:end], labels[start:end])
                balanced_features.append(batch_features)
                balanced_labels.append(batch_labels)
                balanced_file_paths.extend(file_paths[start:end])  # New line to extend file paths
            except ValueError as e:
                print(f"[WARNING] Skipping batch {i} due to insufficient samples: {e}.")
        else:
            print(f"[WARNING] Skipping batch {i} due to only one class present.")

    # Concatenate all balanced features and labels
    balanced_features = np.vstack(balanced_features)
    balanced_labels = np.hstack(balanced_labels)

    balanced_images = balanced_features[:, :images.shape[1] * images.shape[2] * images.shape[3]].reshape(-1, images.shape[1], images.shape[2], images.shape[3])
    balanced_lbp_features = balanced_features[:, images.shape[1] * images.shape[2] * images.shape[3]:]

    return balanced_images, balanced_lbp_features, balanced_labels, balanced_file_paths  # Updated return statement

def train_model_with_lbp_fixed_params(input_images_dir, preprocessed_images_dir, model_save_path):
    start_time = time.time()
    print("[INFO] Starting training process...")
    remove_and_recreate_dir(preprocessed_images_dir)

    print("[INFO] Preprocessing images...")
    preprocess_images(input_images_dir, preprocessed_images_dir)
    print("[INFO] Finished preprocessing images.")

    print("[INFO] Loading images and extracting LBP features...")
    images, lbp_features, labels, label_dict, file_paths = load_images_with_lbp(preprocessed_images_dir)
    print("[INFO] Finished loading images and extracting LBP features.")

    print("[INFO] Balancing classes...")
    balanced_images, balanced_lbp_features, balanced_labels, balanced_file_paths = balance_classes(images, lbp_features, labels, file_paths)
    print("[INFO] Finished balancing classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Extracting ResNet-50 features...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet_features = extract_resnet_features(balanced_images, resnet, device)
    print("[INFO] Finished extracting ResNet-50 features.")

    print("[INFO] Combining features...")
    combined_features = np.hstack((resnet_features, balanced_lbp_features))
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    combined_features = combined_features.astype(np.float32)  # Ensure data is float32
    balanced_labels = balanced_labels.astype(np.int64)  # Ensure labels are of type Long
    print("[INFO] Finished combining and scaling features.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fixed parameters for the model
    fixed_params = {
        'lr': 1e-5,
        'batch_size': 64
    }

    net = NeuralNetClassifier(
        module=LogoClassifierEnsemble,
        module__input_dim=combined_features.shape[1],
        module__output_dim=len(label_dict),
        max_epochs=200,
        lr=fixed_params['lr'],
        optimizer=torch.optim.Adam,
        criterion=torch.nn.CrossEntropyLoss,
        iterator_train__shuffle=True,
        device=device,
        batch_size=fixed_params['batch_size'],
        callbacks=[
            EarlyStopping(patience=5),
            LRScheduler(policy='StepLR', step_size=7, gamma=0.1),
            EpochScoring(scoring='accuracy', lower_is_better=False, on_train=True),
            EpochScoring(scoring='accuracy', lower_is_better=False)
        ]
    )

    scores = []
    for train_idx, val_idx in skf.split(combined_features, balanced_labels):
        X_train, X_val = combined_features[train_idx], combined_features[val_idx]
        y_train, y_val = balanced_labels[train_idx], balanced_labels[val_idx]
        net.fit(X_train, y_train)
        y_pred = net.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    print(f"[INFO] Cross-validation scores with fixed parameters: {scores}")

    best_net = net

    torch.save(best_net.module_.state_dict(), model_save_path)
    joblib.dump(label_dict, 'label_dict.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"[INFO] Model and components saved to {model_save_path} and scaler.pkl")

    print("[INFO] Evaluating final model...")
    y_pred = best_net.predict(combined_features)
    accuracy = accuracy_score(balanced_labels, y_pred)
    precision = precision_score(balanced_labels, y_pred, average='macro', zero_division=0)
    recall = recall_score(balanced_labels, y_pred, average='macro')
    f1 = f1_score(balanced_labels, y_pred, average='macro')
    cm = confusion_matrix(balanced_labels, y_pred)
    specificity = calculate_specificity(cm)

    print(f"\n")
    print(f"[INFO] Evaluation Metrics")
    print(f"[INFO] ==================")
    print(f"[INFO] Accuracy: {accuracy}")
    print(f"[INFO] Precision: {precision}")
    print(f"[INFO] Recall: {recall}")
    print(f"[INFO] Specificity: {specificity}")
    print(f"[INFO] F1 Score: {f1}")
    print(f"[INFO] Confusion Matrix:\n{cm}")

    end_time = time.time()
    print(f"\n[INFO] Total time taken: {end_time - start_time} seconds")

    return best_net, label_dict

if __name__ == "__main__":
    input_images_dir = 'dataset/train'
    preprocessed_images_dir = 'preprocessed_dataset'
    model_save_path = 'logodetection.pth'

    classifier, label_dict = train_model_with_lbp_fixed_params(input_images_dir, preprocessed_images_dir, model_save_path)

# Cross validation with GridSearch for hyperparameters tuning.
'''
def train_model_with_lbp_and_grid_search(input_images_dir, preprocessed_images_dir, model_save_path):
    start_time = time.time()
    print("Starting training process...")
    remove_and_recreate_dir(preprocessed_images_dir)

    print("Preprocessing images...")
    preprocess_images(input_images_dir, preprocessed_images_dir)
    print("Finished preprocessing images.")

    print("Loading images and extracting LBP features...")
    images, lbp_features, labels, label_dict, file_paths = load_images_with_lbp(preprocessed_images_dir)
    print("Finished loading images and extracting LBP features.")

    print("Balancing classes...")
    balanced_images, balanced_lbp_features, balanced_labels, balanced_file_paths = balance_classes(images, lbp_features, labels, file_paths)
    print("Finished balancing classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Extracting ResNet-50 features...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet_features = extract_resnet_features(balanced_images, resnet, device)
    print("Finished extracting ResNet-50 features.")

    print("Combining features...")
    combined_features = np.hstack((resnet_features, balanced_lbp_features))
    scaler = StandardScaler()
    combined_features = scaler.fit_transform(combined_features)
    combined_features = combined_features.astype(np.float32)  # Ensure data is float32
    balanced_labels = balanced_labels.astype(np.int64)  # Ensure labels are of type Long
    print("Finished combining and scaling features.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(params):
        print(f"Evaluating params: {params}")
        net = NeuralNetClassifier(
            module=LogoClassifierEnsemble,
            module__input_dim=combined_features.shape[1],
            module__output_dim=len(label_dict),
            max_epochs=20,
            lr=params['lr'],
            optimizer=torch.optim.Adam,
            criterion=torch.nn.CrossEntropyLoss,
            iterator_train__shuffle=True,
            device=device,
            batch_size=params['batch_size'],
            callbacks=[EarlyStopping(patience=5), LRScheduler(policy='StepLR', step_size=7, gamma=0.1)]
        )
        scores = []
        for train_idx, val_idx in skf.split(combined_features, balanced_labels):
            X_train, X_val = combined_features[train_idx], combined_features[val_idx]
            y_train, y_val = balanced_labels[train_idx], balanced_labels[val_idx]
            net.fit(X_train, y_train)
            y_pred = net.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
        return -np.mean(scores)

    search_space = {
        'lr': [1e-5, 1e-4, 1e-3, 1e-2],
        'batch_size': [16, 32, 64, 128],
    }

    print("Starting Grid Search optimization...")
    grid_search = GridSearchCV(
        estimator=NeuralNetClassifier(
            LogoClassifierEnsemble,
            module__input_dim=combined_features.shape[1],
            module__output_dim=len(label_dict),
            device=device
        ),
        param_grid=search_space,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(combined_features, balanced_labels)
    print("Finished Grid Search optimization.")

    print(f"Best parameters found: {grid_search.best_params_}")
    best_net = grid_search.best_estimator_

    torch.save(best_net.module_.state_dict(), model_save_path)
    joblib.dump(label_dict, 'label_dict.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"Model and components saved to {model_save_path} and scaler.pkl")

    print("Evaluating final model...")
    y_pred = best_net.predict(combined_features)
    accuracy = accuracy_score(balanced_labels, y_pred)
    precision = precision_score(balanced_labels, y_pred, average='macro', zero_division=0)
    recall = recall_score(balanced_labels, y_pred, average='macro')
    f1 = f1_score(balanced_labels, y_pred, average='macro')
    cm = confusion_matrix(balanced_labels, y_pred)
    specificity = calculate_specificity(cm)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Specificity: {specificity}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")

    incorrect_indices = np.where(y_pred != balanced_labels)[0]  # New line to identify incorrect predictions
    print(f"Length of balanced_file_paths: {len(balanced_file_paths)}")  # Debugging prints
    print(f"Length of combined_features: {len(combined_features)}")  # Debugging prints
    print(f"Length of balanced_labels: {len(balanced_labels)}")  # Debugging prints
    print(f"Incorrect indices: {incorrect_indices}")  # Debugging prints
    poor_quality_samples = [balanced_file_paths[i] for i in incorrect_indices]  # New line to get file paths of incorrect predictions

    with open('poor_quality_samples.txt', 'w') as f:  # New block to write poor-quality samples to a file
        for sample in poor_quality_samples:
            f.write(f"{sample}\n")

    print(f"Poor quality samples saved to poor_quality_samples.txt")  # New line for logging

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return best_net, label_dict

if __name__ == "__main__":
    input_images_dir = 'dataset/train'
    preprocessed_images_dir = 'preprocessed_dataset'
    model_save_path = 'classifier_logo_detection.pth'

    classifier, label_dict = train_model_with_lbp_and_grid_search(input_images_dir, preprocessed_images_dir,
                                                                    model_save_path)
'''
