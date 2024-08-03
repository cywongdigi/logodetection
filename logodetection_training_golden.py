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
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from skimage.feature import local_binary_pattern
import joblib
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
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
                    print(f"Warning: Failed to read image {image_path}")
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
    current_label = 0
    for subdir, _, files in os.walk(directory):
        for filename in files:
            if (filename.endswith(('.jpg', '.jpeg', '.png'))):
                image_path = os.path.join(subdir, filename)
                image = cv2.imread(image_path)
                if (image is None):
                    print(f"Warning: Failed to read image {image_path}")
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
    return np.array(images), np.array(lbp_features), np.array(labels), label_dict


def calculate_specificity(cm):
    tn = np.diag(cm).sum() - cm.sum(axis=1)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity = tn / (tn + fp)
    return specificity.mean()


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
        if (self.transform):
            image = self.transform(image)
        return image, lbp_feature, self.labels[idx]


class LogoClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogoClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
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


def balance_classes(images, lbp_features, labels):
    smote = SMOTE()
    combined_features = np.hstack((images.reshape(images.shape[0], -1), lbp_features))
    balanced_features, balanced_labels = smote.fit_resample(combined_features, labels)
    balanced_images = balanced_features[:, :images.shape[1] * images.shape[2] * images.shape[3]].reshape(-1,
                                                                                                         images.shape[
                                                                                                             1],
                                                                                                         images.shape[
                                                                                                             2],
                                                                                                         images.shape[
                                                                                                             3])
    balanced_lbp_features = balanced_features[:, images.shape[1] * images.shape[2] * images.shape[3]:]
    return balanced_images, balanced_lbp_features, balanced_labels


def train_model_with_lbp_and_bayesian_optimization(input_images_dir, preprocessed_images_dir, model_save_path):
    start_time = time.time()
    print("Starting training process...")
    remove_and_recreate_dir(preprocessed_images_dir)

    print("Preprocessing images...")
    preprocess_images(input_images_dir, preprocessed_images_dir)
    print("Finished preprocessing images.")

    print("Loading images and extracting LBP features...")
    images, lbp_features, labels, label_dict = load_images_with_lbp(preprocessed_images_dir)
    print("Finished loading images and extracting LBP features.")

    print("Balancing classes...")
    balanced_images, balanced_lbp_features, balanced_labels = balance_classes(images, lbp_features, labels)
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

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(params):
        print(f"Evaluating params: {params}")
        net = NeuralNetClassifier(
            module=LogoClassifier,
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
        'lr': Real(1e-5, 1e-2, prior='log-uniform'),
        'batch_size': Integer(16, 128),
    }

    print("Starting Bayesian optimization...")
    bayes_search = BayesSearchCV(
        estimator=NeuralNetClassifier(
            LogoClassifier,
            module__input_dim=combined_features.shape[1],
            module__output_dim=len(label_dict),
            device=device
        ),
        search_spaces=search_space,
        n_iter=10,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        refit=True,
        optimizer_kwargs={'base_estimator': 'GP'},
        random_state=42,
    )
    bayes_search.fit(combined_features, balanced_labels)
    print("Finished Bayesian optimization.")

    print(f"Best parameters found: {bayes_search.best_params_}")
    best_net = bayes_search.best_estimator_

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

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return best_net, label_dict


if __name__ == "__main__":
    input_images_dir = 'dataset/train'
    preprocessed_images_dir = 'preprocessed_dataset'
    model_save_path = 'classifier_logo_detection.pth'

    classifier, label_dict = train_model_with_lbp_and_bayesian_optimization(input_images_dir, preprocessed_images_dir,
                                                                            model_save_path)
