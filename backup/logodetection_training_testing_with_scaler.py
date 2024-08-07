import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
from PIL import Image

# Define the target size
TARGET_SIZE = 224


# Define the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def set_paths():
    if 'COLAB_GPU' in os.environ:  # Google Colab environment variable
        print("[INFO] Running on Google Colab")
        input_images_dir = '/drive/My Drive/Colab Notebooks/dataset/train'
        test_images_dir = '/drive/My Drive/Colab Notebooks/dataset/test'
        preprocessed_images_dir = '/drive/My Drive/Colab Notebooks/preprocessed_dataset'
        preprocessed_test_images_dir = '/drive/My Drive/Colab Notebooks/preprocessed_test_dataset'
    else:
        print("[INFO] Running on local environment (PyCharm)")
        input_images_dir = 'dataset/train'
        test_images_dir = 'dataset/train'
        preprocessed_images_dir = 'preprocessed_dataset'
        preprocessed_test_images_dir = 'preprocessed_test_dataset'

    return input_images_dir, test_images_dir, preprocessed_images_dir, preprocessed_test_images_dir


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


# Define Data Augmentation and Normalization Transforms
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Rotate by +/- 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


normalization_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


def color_correction(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    corrected_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return corrected_image


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def smooth_image(image):
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed


def preprocess_images(input_dir, output_dir, augment=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                image = cv2.imread(file_path)
                image = resize_and_pad_image(image)
                image = color_correction(image)
                image = sharpen_image(image)
                image = smooth_image(image)

                # Apply augmentation transforms by default, use normalization for test data if augment is False
                if augment:
                    image = augmentation_transforms(image)
                else:
                    image = normalization_transform(image)

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
    # Load the ResNet model and remove the final classification layer
    resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet_model.fc = nn.Identity()  # Remove the fully connected layer
    resnet_model.eval()
    resnet_model = resnet_model.to(device)

    processed_data = []
    for img in data:
        if model_type == 'lbp':
            features = extract_lbp_features(img)
        elif model_type == 'resnet50_lbp':
            # Convert the NumPy array to a PIL image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Normalize the image for ResNet
            resnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Ensure resizing
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = resnet_transform(img_pil).unsqueeze(0).to(device)

            # Extract ResNet features
            with torch.no_grad():
                resnet_features = resnet_model(img_tensor).cpu().numpy().flatten()  # Move back to CPU to process with NumPy

            # Extract LBP features
            lbp_features = extract_lbp_features(img).reshape(-1)

            # Print individual feature shapes for debugging
            # print(f"ResNet Features Shape: {resnet_features.shape}")
            # print(f"LBP Features Shape: {lbp_features.shape}")

            # Combine features
            combined_features = np.hstack((resnet_features, lbp_features))
            features = combined_features
        elif model_type == 'resnet50':
            # Convert the NumPy array to a PIL image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Normalize the image for ResNet
            resnet_transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Ensure resizing
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = resnet_transform(img_pil).unsqueeze(0).to(device)  # Apply transformations and move to device

            # Extract ResNet features
            with torch.no_grad():  # Disable gradient calculation
                resnet_features = resnet_model(img_tensor).cpu().numpy().flatten()  # Move back to CPU to process with NumPy

            features = resnet_features
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


class CombinedDataset(Dataset):
    def __init__(self, resnet_features, lbp_features, labels):
        self.resnet_features = torch.tensor(resnet_features, dtype=torch.float32)
        self.lbp_features = torch.tensor(lbp_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        resnet_feature = self.resnet_features[idx]
        lbp_feature = self.lbp_features[idx]
        label = self.labels[idx]
        return resnet_feature, lbp_feature, label


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

            for batch in dataloaders[phase]:
                if model_type == 'resnet50_lbp':
                    resnet_inputs, lbp_inputs, labels = batch
                    resnet_inputs = resnet_inputs.to(device)
                    lbp_inputs = lbp_inputs.to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if model_type == 'lbp' or model_type == 'resnet50':
                        outputs = model(inputs)
                    else:
                        resnet_outputs = model[0](resnet_inputs).view(resnet_inputs.size(0), -1)  # Forward pass through ResNet and flatten
                        combined_inputs = torch.cat((resnet_outputs, lbp_inputs), dim=1)
                        outputs = model[1](combined_inputs)  # Forward pass through combined part

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)
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
        for batch in dataloader:
            if model_type == 'lbp':
                inputs, labels = batch
                inputs = inputs.to(device)
            elif model_type == 'resnet50':
                inputs, labels = batch
                inputs = inputs.to(device)
            elif model_type == 'resnet50_lbp':
                resnet_inputs, lbp_inputs, labels = batch
                resnet_inputs = resnet_inputs.to(device)
                lbp_inputs = lbp_inputs.to(device)

            labels = labels.to(device)

            if model_type == 'lbp' or model_type == 'resnet50':
                outputs = model(inputs)
            else:
                resnet_outputs = model[0](resnet_inputs).view(resnet_inputs.size(0), -1)
                combined_inputs = torch.cat((resnet_outputs, lbp_inputs), dim=1)
                outputs = model[1](combined_inputs)  # Forward pass through combined part

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
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


if __name__ == "__main__":
    start_time = time.time()

    input_images_dir, test_images_dir, preprocessed_images_dir, preprocessed_test_images_dir = set_paths()

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a model for logo detection.")
    parser.add_argument('--model_type', type=str, choices=['lbp', 'resnet50', 'resnet50_lbp'], required=True, help="Specify the model type to use: 'lbp', 'resnet50', or 'resnet50_lbp'.")
    args = parser.parse_args()

    # Use the model_type argument
    model_type = args.model_type # Example model type lbp, resnet50, resnet50_lbp

    print(f"[INFO] model_type12: {model_type}")

    # Preprocess data
    print("[INFO] Preprocessing images...")
    # if 'COLAB_GPU' not in os.environ:
    #     preprocess_images(input_images_dir, preprocessed_images_dir, augment=True)

    # Load and preprocess data based on model type
    print("[INFO] Loading images and extracting features...")
    images, lbp_features, labels, label_dict, file_paths = load_images_with_lbp(preprocessed_images_dir)

    # Remove extraneous labels if necessary
    if 'preprocessed_dataset' in label_dict:
        del label_dict['preprocessed_dataset']

    # Split data
    print("[INFO] Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"[INFO] Split into {len(X_train)} training and {len(X_test)} testing samples.")

    train_data = preprocess_data(X_train, model_type)
    test_data = preprocess_data(X_test, model_type)
    print(f"[INFO] Preprocessed data for model type {model_type}.")

    # Check and handle tensor shapes directly in the main script
    if model_type == 'lbp':
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
    elif model_type == 'resnet50_lbp':
        combined_train_data = np.array(train_data)
        combined_test_data = np.array(test_data)
    else:
        train_data = torch.tensor(train_data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to NCHW
        test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)  # Convert to NCHW

    # Fit the scaler on the training data
    scaler = StandardScaler()
    if model_type == 'resnet50_lbp':
        scaler.fit(combined_train_data)
        print(f"Expected number of features: {combined_train_data.shape[1]}")
    else:
        scaler.fit(train_data)

    # if model_type == 'lbp':
    #     train_dataset = TensorDataset(train_data, torch.tensor(y_train, dtype=torch.long))
    #     test_dataset = TensorDataset(test_data, torch.tensor(y_test, dtype=torch.long))
    # elif model_type == 'resnet50_lbp':
    #     resnet_features_train = np.array([t[:-26] for t in combined_train_data])
    #     lbp_features_train = np.array([t[-26:] for t in combined_train_data])
    #     resnet_features_test = np.array([t[:-26] for t in combined_test_data])
    #     lbp_features_test = np.array([t[-26:] for t in combined_test_data])
    #
    #     train_dataset = CombinedDataset(resnet_features_train, lbp_features_train, y_train)
    #     test_dataset = CombinedDataset(resnet_features_test, lbp_features_test, y_test)
    # else:
    #     train_dataset = TensorDataset(train_data, torch.tensor(y_train, dtype=torch.long))
    #     test_dataset = TensorDataset(test_data, torch.tensor(y_test, dtype=torch.long))
    #
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # dataloaders = {'train': train_loader, 'val': test_loader}
    #
    # # Build model
    # num_classes = len(label_dict)
    # model = build_model(model_type, num_classes)
    # model = model.to(device)
    #
    # # Define loss criterion and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #
    # # Define early stopping and learning rate scheduler
    # early_stopping = CustomEarlyStopping(patience=5)
    # lr_scheduler = CustomLRScheduler(optimizer, step_size=7, gamma=0.1)
    # callbacks = [early_stopping, lr_scheduler]
    #
    # # Train model
    # print("[INFO] Training model...")
    # model = train_model(model, dataloaders, criterion, optimizer, callbacks=callbacks, num_epochs=100)
    #
    # # Evaluate model
    # print("[INFO] Evaluating model...")
    # evaluate_model(model, test_loader)
    #
    # # Save model, scaler, and label dictionary
    # print("[INFO] Saving model, scaler, and label dictionary...")
    # save_model_and_scaler(model, scaler, model_type, label_dict)
    #
    # # Load the model and label dictionary for testing
    # print(f"[INFO] Loading model and label dictionary for {model_type}...")
    # model.load_state_dict(torch.load(f'{model_type}_model.pth', weights_only=True))
    # label_dict = joblib.load(f'{model_type}_label_dict.pkl')
    #
    # # Preprocess test images
    # print("[INFO] Preprocessing test images...")
    # if 'COLAB_GPU' not in os.environ:
    #     preprocess_images(test_images_dir, preprocessed_test_images_dir, augment=False)
    #
    # # Load and preprocess test data
    # print("[INFO] Loading test images and extracting features...")
    # test_images, test_lbp_features, test_labels, _, test_file_paths = load_images_with_lbp('preprocessed_test_dataset')
    # test_data = preprocess_data(test_images, model_type)
    #
    # # Check and handle tensor shapes directly in the main script
    # if model_type == 'lbp':
    #     test_data = torch.tensor(test_data, dtype=torch.float32)
    #     test_dataset = TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.long))
    # elif model_type == 'resnet50_lbp':
    #     resnet_features_test = np.array([t[:-26].reshape(3, 224, 224) for t in test_data])
    #     lbp_features_test = np.array([t[-26:] for t in test_data])
    #     combined_test_data = np.hstack([resnet_features_test.reshape(len(resnet_features_test), -1), lbp_features_test])
    #     test_dataset = CombinedDataset(resnet_features_test, lbp_features_test, test_labels)
    # else:
    #     test_data = torch.tensor(test_data, dtype=torch.float32).permute(0, 3, 1, 2)
    #     test_dataset = TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.long))
    #
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # # Evaluate model on test data
    # print("[INFO] Evaluating model on test data...")
    # evaluate_model(model, test_loader)
