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

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def smooth_image(image):
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    return smoothed

def correct_color(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(result))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    result = cv2.merge(tuple(lab_planes))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

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
        self.root.geometry("1600x900")
        self.create_frames()
        self.create_buttons()
        self.model = self.load_model()
        self.current_image = None

    def create_frames(self):
        self.window1_frame = tk.LabelFrame(self.root, text="Custom Image", width=800, height=600, bd=1, relief="solid")
        self.window1_frame.place(x=10, y=10)
        self.window1 = tk.Label(self.window1_frame)
        self.window1.pack(expand=True)

        self.window2_frame = tk.LabelFrame(self.root, text="Resizing & Padding", width=224, height=224, bd=1,
                                           relief="solid")
        self.window2_frame.place(x=10, y=640)
        self.window2 = tk.Label(self.window2_frame)
        self.window2.pack(expand=True)

        self.window3_frame = tk.LabelFrame(self.root, text="Color Correction", width=224, height=224, bd=1,
                                           relief="solid")
        self.window3_frame.place(x=244, y=640)
        self.window3 = tk.Label(self.window3_frame)
        self.window3.pack(expand=True)

        self.window4_frame = tk.LabelFrame(self.root, text="Sharpening", width=224, height=224, bd=1, relief="solid")
        self.window4_frame.place(x=478, y=640)
        self.window4 = tk.Label(self.window4_frame)
        self.window4.pack(expand=True)

        self.window5_frame = tk.LabelFrame(self.root, text="Smoothing", width=224, height=224, bd=1, relief="solid")
        self.window5_frame.place(x=712, y=640)
        self.window5 = tk.Label(self.window5_frame)
        self.window5.pack(expand=True)

        self.logo_detection_result_frame = tk.LabelFrame(self.root, text="Logo Classification", width=224, height=224,
                                                         bd=1, relief="solid")
        self.logo_detection_result_frame.place(x=946, y=640)
        self.logo_detection_result = tk.Label(self.logo_detection_result_frame)
        self.logo_detection_result.pack(expand=True)

        self.red_hist_frame = tk.LabelFrame(self.root, text="Red Color Histogram", width=300, height=100, bd=1,
                                            relief="solid")
        self.red_hist_frame.place(x=820, y=10)
        self.red_hist = tk.Label(self.red_hist_frame)
        self.red_hist.pack(expand=True)

        self.green_hist_frame = tk.LabelFrame(self.root, text="Green Color Histogram", width=300, height=100, bd=1,
                                              relief="solid")
        self.green_hist_frame.place(x=820, y=260)
        self.green_hist = tk.Label(self.green_hist_frame)
        self.green_hist.pack(expand=True)

        self.blue_hist_frame = tk.LabelFrame(self.root, text="Blue Color Histogram", width=300, height=100, bd=1,
                                             relief="solid")
        self.blue_hist_frame.place(x=820, y=510)
        self.blue_hist = tk.Label(self.blue_hist_frame)
        self.blue_hist.pack(expand=True)

    def create_buttons(self):
        self.upload_button = tk.Button(self.root, text="Upload Custom Image", command=self.upload_image, width=20)
        self.upload_button.place(x=10, y=854)

        self.classify_button = tk.Button(self.root, text="Classify Logo", command=self.classify_logo, width=20)
        self.classify_button.place(x=780, y=854)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.quit, width=20)
        self.exit_button.place(x=1000, y=854)

    def load_model(self):
        model = LogoClassifierEnsemble(input_dim=2074, output_dim=20)
        state_dict = torch.load("/mnt/data/logodetection.pth", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

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
            self.display_image_with_roi(image, self.window1, file_path, original_width, original_height)

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
        image_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        window.config(image=image_tk)
        window.image = image_tk

    def classify_logo_from_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        image = Image.fromarray(image)  # Convert OpenCV image back to PIL format
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            self.logo_detection_result.config(text=f"Detected Logo: {predicted.item()}")

    def classify_logo(self):
        if self.current_image is not None:
            features = self.extract_features(self.current_image)
            output = self.model(features)
            _, predicted = torch.max(output, 1)
            self.logo_detection_result.config(text=f"Detected Logo: {predicted.item()}")
        else:
            self.logo_detection_result.config(text="No image uploaded.")

    def extract_features(self, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
        ])
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image

if __name__ == "__main__":
    root = tk.Tk()
    app = LogoDetectionGUI(root)
    root.mainloop()
