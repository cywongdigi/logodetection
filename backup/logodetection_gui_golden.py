import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from google.cloud import vision
import os
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import models
from skimage.feature import local_binary_pattern
import joblib  # For loading the label dictionary
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "genuine-vent-430507-b0-5293ed9cf15e.json"

TARGET_SIZE = 224


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


def correct_color(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(result))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    result = cv2.merge(tuple(lab_planes))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def smooth_image(image):
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed


# This should be consistent in both scripts
def normalize_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def extract_lbp_features(image, num_points=24, radius=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)  # Convert to uint8 if not already
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_resnet_features(image, resnet, device):
    resnet.eval()
    resnet = resnet.to(device)
    image = image.to(device).float()
    with torch.no_grad():
        feature = resnet(image).cpu().numpy().flatten()
    return feature


class LogoClassifierEnsemble(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogoClassifierEnsemble, self).__init__()
        self.fc1 = nn.Linear(2074, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 20)

        self.fc3 = nn.Linear(2074, 256)
        self.fc4 = nn.Linear(256, 20)

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


class LogoDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Non-ODL Group 4 - Automated Detection of Logos")
        self.root.geometry("1184x926")
        self.create_frames()
        self.create_buttons()
        self.model = self.load_model()
        self.feature_extractor = self.load_feature_extractor()
        self.label_dict = self.load_label_dict()  # Load the label dictionary
        self.scaler = self.load_scaler()  # Load the scaler
        self.inverted_label_dict = {v: k for k, v in self.label_dict.items()}  # Invert the dictionary
        self.current_image = None

    def create_frames(self):
        self.window1_frame = tk.LabelFrame(self.root, text="Custom Image", width=802, height=618, bd=1, relief="solid")
        self.window1_frame.place(x=10, y=10)
        self.window1_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.window1 = tk.Label(self.window1_frame)
        self.window1.place(x=0, y=0, width=800, height=600)  # Ensure the label fills the entire frame

        self.window2_frame = tk.LabelFrame(self.root, text="Resizing & Padding", width=226, height=242, bd=1, relief="solid")
        self.window2_frame.place(x=10, y=638)
        self.window2_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.window2 = tk.Label(self.window2_frame)
        self.window2.place(x=0, y=0, width=224, height=224)  # Ensure the label fills the entire frame

        self.window3_frame = tk.LabelFrame(self.root, text="Color Correction", width=226, height=242, bd=1, relief="solid")
        self.window3_frame.place(x=246, y=638)
        self.window3_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.window3 = tk.Label(self.window3_frame)
        self.window3.place(x=0, y=0, width=224, height=224)  # Ensure the label fills the entire frame

        self.window4_frame = tk.LabelFrame(self.root, text="Sharpening", width=226, height=242, bd=1, relief="solid")
        self.window4_frame.place(x=480, y=638)
        self.window4_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.window4 = tk.Label(self.window4_frame)
        self.window4.place(x=0, y=0, width=224, height=224)  # Ensure the label fills the entire frame

        self.window5_frame = tk.LabelFrame(self.root, text="Smoothing", width=226, height=242, bd=1, relief="solid")
        self.window5_frame.place(x=714, y=638)
        self.window5_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.window5 = tk.Label(self.window5_frame)
        self.window5.place(x=0, y=0, width=224, height=224)  # Ensure the label fills the entire frame

        self.logo_detection_result_frame = tk.LabelFrame(self.root, text="Logo Classification", width=226, height=242, bd=1, relief="solid")
        self.logo_detection_result_frame.place(x=948, y=638)
        self.logo_detection_result_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget
        self.logo_detection_result = tk.Label(self.logo_detection_result_frame, font=("Arial", 18))
        self.logo_detection_result.place(x=0, y=0, width=224, height=224)  # Ensure the label fills the entire frame

        self.rgb_hist_frame = tk.LabelFrame(self.root, text="RGB Histogram", width=352, height=618, bd=1, relief="solid")
        self.rgb_hist_frame.place(x=822, y=10)
        self.rgb_hist_frame.pack_propagate(False)  # Prevent the frame from resizing to fit the child widget

    def create_buttons(self):
        self.upload_button = tk.Button(self.root, text="Upload Custom Image", command=self.upload_image, width=20)
        self.upload_button.place(x=10, y=890)

        self.classify_button = tk.Button(self.root, text="Classify Logo", command=self.classify_logo, width=20)
        self.classify_button.place(x=480, y=890)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit, width=20)
        self.exit_button.place(x=948, y=890)

    def load_model(self):
        model = LogoClassifierEnsemble(input_dim=2074, output_dim=20)
        state_dict = torch.load("logodetection.pth", map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def load_feature_extractor(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
        resnet.eval()
        return resnet

    def load_label_dict(self):
        label_dict = joblib.load("label_dict.pkl")  # Load the label dictionary from the file
        for index, label in label_dict.items():
            print(f"Index: {index}, Label: {label}")  # Print all the labels and their indices
        return label_dict

    def load_scaler(self):
        return joblib.load("scaler.pkl")  # Load the scaler from the file

    def extract_features(self, image):
        # Normalize the image
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet_features = extract_resnet_features(image, self.feature_extractor, device)
        lbp_features = extract_lbp_features(cv2.cvtColor(np.array(image.squeeze().permute(1, 2, 0)), cv2.COLOR_RGB2BGR))
        lbp_features = lbp_features.reshape(-1)  # Ensure LBP features have 1 dimension
        combined_features = np.hstack((resnet_features, lbp_features))

        # Apply the scaler to the combined features
        combined_features = self.scaler.transform([combined_features]).astype(np.float32)
        combined_features = torch.tensor(combined_features).float().unsqueeze(0)  # Add batch dimension
        return combined_features

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            original_width, original_height = image.size
            if image.width > 800 or image.height > 600:
                image = image.resize((800, 600), Image.Resampling.LANCZOS)
            else:
                image = ImageOps.pad(image, (800, 600), method=Image.Resampling.LANCZOS, color=(255, 255, 255))
            self.current_image = image.copy()  # Make a copy to keep the original image unchanged
            normalized_image = np.array(image).astype(np.float32) / 255.0  # Scale to [0, 1]
            self.display_image_with_roi(image, self.window1, file_path, original_width, original_height)
            self.display_histograms(image)  # Display histograms
            # self.classify_logo(normalized_image)  # Automatically classify logo after uploading the image

    def display_image_with_roi(self, image, window, file_path, original_width, original_height):
        modified_image, rois = self.draw_roi_on_image(file_path, image, original_width, original_height)
        image_tk = ImageTk.PhotoImage(modified_image)
        window.config(image=image_tk)
        window.image = image_tk

        for roi in rois:
            self.process_and_display_roi(roi)

    def draw_roi_on_image(self, file_path, image, original_width, original_height):
        client = vision.ImageAnnotatorClient()
        with open(file_path, "rb") as image_file:
            content = image_file.read()
        image_annotator = vision.Image(content=content)
        response = client.logo_detection(image=image_annotator)
        logos = response.logo_annotations

        draw = ImageDraw.Draw(image)
        rois = []
        for logo in logos:
            vertices = [(vertex.x * 800 / original_width, vertex.y * 600 / original_height) for vertex in
                        logo.bounding_poly.vertices]
            if len(vertices) == 4:
                draw.rectangle([vertices[0], vertices[2]], outline="yellow", width=3)
                roi_box = (vertices[0][0] + 3, vertices[0][1] + 3, vertices[2][0] - 3, vertices[2][1] - 3)
                roi = image.crop(roi_box)
                rois.append((roi, roi_box))

        if response.error.message:
            raise Exception(f"{response.error.message}")

        return image, rois

    def process_and_display_roi(self, roi_data):
        roi, roi_box = roi_data

        # Convert PIL image to OpenCV format
        roi_cv2 = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)

        # Resize and pad ROI
        resized_padded_roi = resize_and_pad_image(roi_cv2)
        self.display_processed_image(resized_padded_roi, self.window2)

        # Color correction
        color_corrected_roi = correct_color(resized_padded_roi)
        self.display_processed_image(color_corrected_roi, self.window3)

        # Sharpen the resized and padded ROI
        sharpened_roi = sharpen_image(color_corrected_roi)
        self.display_processed_image(sharpened_roi, self.window4)

        # Smooth the sharpened ROI
        smoothed_roi = smooth_image(sharpened_roi)
        self.display_processed_image(smoothed_roi, self.window5)

        # Classify the smoothed ROI
        self.classify_logo_from_image(smoothed_roi)

    def display_processed_image(self, image, window):
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a format that Tkinter can use
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Create a label to display the image
        label = tk.Label(window, image=image_tk)
        label.image = image_tk  # Keep a reference to avoid garbage collection
        label.pack()

    def classify_logo_from_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        image = Image.fromarray(image)  # Convert OpenCV image back to PIL format
        features = self.extract_features(image)
        with torch.no_grad():
            output = self.model(features)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()  # Get the predicted index
            brand_label = self.inverted_label_dict.get(predicted_index, "Unknown")  # Get the brand label
            print(f"From Image: Predicted Index: {predicted_index}, Predicted Label: {brand_label}")  # Print the index and label
            self.logo_detection_result.config(text=f"{brand_label}")

    def classify_logo(self, image=None):  # Accept an optional image parameter
        if image is None:
            if self.current_image is not None:
                image = self.current_image
            else:
                self.logo_detection_result.config(text="No image uploaded.")
                return

        features = self.extract_features(image)
        with torch.no_grad():
            output = self.model(features)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()  # Get the predicted index
            brand_label = self.inverted_label_dict.get(predicted_index, "Unknown")  # Get the brand label
            print(f"No Image Predicted Index: {predicted_index}, Predicted Label: {brand_label}")  # Print the index and label
            self.logo_detection_result.config(text=f"{brand_label}")

    def display_histograms(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        r, g, b = image.split()
        fig, axs = plt.subplots(3, 1, figsize=(4, 6))  # Adjust the figure size to 400x600

        for ax, channel, color in zip(axs, [r, g, b], ['red', 'green', 'blue']):
            hist = channel.histogram()
            ax.bar(range(256), hist, color=color)
            ax.set_xlim(0, 255)
            ax.set_ylim(0, max(hist))
            ax.set_title(f'{color.capitalize()} Color Histogram', fontsize=10)  # Set the title font size

        plt.tight_layout()

        # Ensure the frame does not resize based on the content
        self.rgb_hist_frame.pack_propagate(False)

        canvas = FigureCanvasTkAgg(fig, master=self.rgb_hist_frame)
        canvas.get_tk_widget().pack(expand=True, fill='both')
        canvas.draw()
        plt.close(fig)  # Close the figure after plotting


if __name__ == "__main__":
    root = tk.Tk()
    app = LogoDetectionGUI(root)
    root.mainloop()
