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
from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.feature import local_binary_pattern
import joblib
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler, EpochScoring
import argparse
from tabulate import tabulate
import time
from torchvision.models import ResNet50_Weights

# Define the target size
TARGET_SIZE = 224

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomEarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True


class CustomLRScheduler:
    def __init__(self, optimizer, step_size=7, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = -1

    def on_epoch_end(self, epoch, logs=None):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.gamma


def normalize_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


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


def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                image = color_correction(image)
                image = resize_and_pad_image(image)
                image = sharpen_image(image)
                image = smooth_image(image)
                image = normalize_image(image)
                relative_path = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                output_path = os.path.join(output_subdir, file)
                image = image.squeeze(0).permute(1, 2, 0).numpy()  # Convert tensor back to NumPy array
                cv2.imwrite(output_path, (image * 255).astype(np.uint8))
    print("[INFO] Finished preprocessing images.")


def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(cv2.convertScaleAbs(image), cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.astype(np.uint8)  # Convert to integer type
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def preprocess_data(data, model_type):
    processed_data = []
    for img in data:
        if model_type == 'lbp':
            features = extract_lbp_features(img)
        elif model_type == 'resnet50_lbp':
            # Normalize the image for ResNet
            resnet_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            resnet_features = resnet_transform(img).unsqueeze(0)

            # Extract LBP features
            lbp_features = extract_lbp_features(img)

            # Combine features
            features = np.hstack((resnet_features.numpy().flatten(), lbp_features))
        else:
            features = img

        processed_data.append(features)

    return np.array(processed_data)


def load_images_with_lbp(directory):
    images = []
    lbp_features = []
    labels = []
    file_paths = []
    label_dict = {}
    label_counter = 0

    for subdir, _, files in os.walk(directory):
        if not os.path.isdir(subdir):
            continue  # Skip if not a directory

        class_name = os.path.basename(subdir)
        if class_name not in label_dict:
            label_dict[class_name] = label_counter
            label_counter += 1

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"[WARNING] Failed to load image: {file_path}")
                    continue
                images.append(image)
                lbp_features.append(extract_lbp_features(image))
                labels.append(label_dict[class_name])
                file_paths.append(file_path)

    return np.array(images), np.array(lbp_features), np.array(labels), label_dict, file_paths


def build_model(model_type, num_classes):
    if model_type == 'lbp':
        model = nn.Sequential(
            nn.Linear(26, 100),  # LBP features size
            nn.ReLU(),
            nn.Linear(100, num_classes)  # Using num_classes
        )
    elif model_type == 'resnet50':
        resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, num_classes)  # Replace the fully connected layer with our own
        model = resnet_model
    elif model_type == 'resnet50_lbp':
        resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Identity()  # Remove the fully connected layer

        model = nn.Sequential(
            resnet_model,
            nn.Linear(num_ftrs + 26, 100),  # Combine ResNet features with LBP
            nn.ReLU(),
            nn.Linear(100, num_classes)  # Using num_classes
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"[INFO] Built {model_type} model.")
    return model


def save_model_and_scaler(model, scaler, model_type, label_dict):
    torch.save(model.state_dict(), f'{model_type}_model.pth')
    joblib.dump(scaler, f'{model_type}_scaler.pkl')
    joblib.dump(label_dict, f'{model_type}_label_dict.pkl')  # Save label dictionary
    print(f"[INFO] Saved {model_type} model, scaler, and label dictionary.")


def train_model(model, dataloaders, criterion, optimizer, callbacks=None, num_epochs=25):
    headers = ["epoch", "train_loss", "train_acc", "valid_loss", "valid_acc", "dur"]
    results = []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.permute(0, 3, 1, 2).to(device)  # Ensure input is in NCHW format
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss, train_acc = epoch_loss, epoch_acc
            else:
                val_loss, val_acc = epoch_loss, epoch_acc

        duration = time.time() - start_time
        results.append([epoch + 1, train_loss, train_acc.item(), val_loss, val_acc.item(), duration])

        # Early Stopping and Learning Rate Scheduling
        if callbacks:
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs={'val_loss': val_loss, 'model': model})

    print(tabulate(results, headers=headers, floatfmt=".4f"))

    return model


def calculate_specificity(cm):
    tn = np.sum(np.diag(cm)) - np.sum(cm, axis=1)
    fp = np.sum(cm, axis=0) - np.diag(cm)
    specificity = np.divide(tn, (tn + fp), out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    return specificity.mean()


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    specificity = calculate_specificity(cm)  # Calculate specificity

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")  # Print specificity
    print(f"Confusion Matrix:\n{cm}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"[INFO] Total run time: {total_time:.2f} seconds")


def balance_classes(images, lbp_features, labels, file_paths, batch_size=100):
    smote = SMOTE(k_neighbors=5)
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

    if len(balanced_features) == 0:
        raise ValueError("[ERROR] No batches could be resampled. Check your dataset and batch sizes.")

    # Concatenate all balanced features and labels
    balanced_features = np.vstack(balanced_features)
    balanced_labels = np.hstack(balanced_labels)

    balanced_images = balanced_features[:, :images.shape[1] * images.shape[2] * images.shape[3]].reshape(-1,
                                                                                                         images.shape[
                                                                                                             1],
                                                                                                         images.shape[
                                                                                                             2],
                                                                                                         images.shape[
                                                                                                             3])
    balanced_lbp_features = balanced_features[:, images.shape[1] * images.shape[2] * images.shape[3]:]

    return balanced_images, balanced_lbp_features, balanced_labels, balanced_file_paths  # Updated return statement


def print_class_distribution(labels, label_dict):
    label_counts = {label: 0 for label in label_dict.values()}
    for label in labels:
        label_counts[label] += 1
    for label, count in label_counts.items():
        print(f"Class {label} ({list(label_dict.keys())[list(label_dict.values()).index(label)]}): {count} samples")


def visualize_images(images, labels, label_dict, num_images=5):
    for i in range(num_images):
        img = images[i]
        label = labels[i]
        label_name = list(label_dict.keys())[list(label_dict.values()).index(label)]
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Class: {label_name}")
        plt.show()


if __name__ == "__main__":
    start_time = time.time()
    input_images_dir = 'dataset/train'
    preprocessed_images_dir = 'preprocessed_dataset'
    model_save_path = 'logodetection.pth'

    model_type = 'resnet50'  # Example model type lbp, resnet50, resnet50_lbp

    # Preprocess data
    print("[INFO] Preprocessing images...")
    preprocess_images(input_images_dir, preprocessed_images_dir)

    # Load and preprocess data based on model type
    print("[INFO] Loading images and extracting features...")
    images, lbp_features, labels, label_dict, file_paths = load_images_with_lbp(preprocessed_images_dir)
    # print_class_distribution(labels, label_dict)  # Print training class distribution

    # Balance the classes
    # print("[INFO] Balancing classes...")
    # images, lbp_features, labels, file_paths = balance_classes(images, lbp_features, labels, file_paths)
    # print_class_distribution(labels, label_dict)  # Print training class distribution

    # Split data
    print("[INFO] Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    train_data = preprocess_data(X_train, model_type)
    test_data = preprocess_data(X_test, model_type)

    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    dataloaders = {'train': train_loader, 'val': test_loader}

    # Build model
    num_classes = len(label_dict)
    model = build_model(model_type, num_classes)
    model = model.to(device)

    # Define loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define early stopping and learning rate scheduler
    early_stopping = CustomEarlyStopping(patience=5)
    lr_scheduler = CustomLRScheduler(optimizer, step_size=7, gamma=0.1)
    callbacks = [early_stopping, lr_scheduler]

    # Set number of epochs based on model type
    if model_type == 'lbp':
        num_epochs = 100
    elif model_type == 'resnet50' or model_type == 'resnet50_lbp':
        num_epochs = 20
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train model
    print("[INFO] Training model...")
    model = train_model(model, dataloaders, criterion, optimizer, callbacks=callbacks, num_epochs=num_epochs)

    # Evaluate model
    print("[INFO] Evaluating model...")
    evaluate_model(model, test_loader)

    # Save model, scaler, and label dictionary
    print("[INFO] Saving model, scaler, and label dictionary...")
    save_model_and_scaler(model, StandardScaler(), model_type, label_dict)

    # Load the model and label dictionary for testing
    print(f"[INFO] Loading model and label dictionary for {model_type}...")
    model.load_state_dict(torch.load(f'{model_type}_model.pth', weights_only=True))
    label_dict = joblib.load(f'{model_type}_label_dict.pkl')

    # Preprocess test images
    test_images_dir = 'dataset/test'
    print("[INFO] Preprocessing test images...")
    preprocess_images(test_images_dir, 'preprocessed_test_dataset')

    # Load and preprocess test data
    print("[INFO] Loading test images and extracting features...")
    test_images, test_lbp_features, test_labels, _, test_file_paths = load_images_with_lbp('preprocessed_test_dataset')
    test_data = preprocess_data(test_images, model_type)

    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate model on test data
    print("[INFO] Evaluating model on test data...")
    evaluate_model(model, test_loader)
