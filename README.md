Sure, here's the updated `README.md` file with descriptions for `lbp`, `resnet50`, and `resnet50_lbp`:

---

# Logo Detection System

## Overview
This repository contains a logo detection system that utilizes a combination of traditional image processing techniques and deep learning models to detect and classify logos in images. The system is built using Python and leverages libraries such as OpenCV, PyTorch, and Google Cloud Vision API.

## Repository Structure
```
.
├── logodetection_training_testing.py        # Main Python script for training and testing the logo detection system
├── logodetection_gui.py                     # Main Python script for the logo detection system GUI
├── genuine-vent-430507-b0-5293ed9cf15e.json # Google Cloud Vision API credentials JSON file for ROI detection in uploaded custom image
├── dataset                                  # Logo images for training and testing the logo detection system
├── preprocessed_dataset                     # Preprocessed train dataset used for training the logo detection system
├── preprocessed_test_dataset                # Preprocessed test dataset for evaluating the logo detection system
├── model                                    # Pre-trained model, label dictionary, and scaler for all 3 types of models
├── log                                      # Training and testing log file for all 3 types of models
├── custom_image                             # Custom images used for testing or training the logo detection system
├── gui_image                                # Snippets of the GUI for correct logo classifications
├── backup                                   # Backup files related to the project
└── README.md                                # Provides an overview, setup instructions, and usage guidelines for the logo detection system project
```

## Steps to Run `logodetection_gui.py`

### Prerequisites
Ensure you have Python installed (preferably Python 3.7 or later).

### Python Libraries Required
Install the required libraries using the following command:
```bash
pip install tkinter pillow torch torchvision google-cloud-vision opencv-python-headless numpy scikit-image joblib matplotlib tabulate skorch
```

### Running the GUI
1. Clone the repository:
   ```bash
   git clone https://github.com/cywongdigi/logodetection.git
   cd logodetection
   ```

2. Ensure the Google Cloud Vision API credentials JSON file (`genuine-vent-430507-b0-5293ed9cf15e.json`) is placed in the same directory as `logodetection_gui.py`.

3. Run the GUI script:
   ```bash
   python logodetection_gui.py
   ```

### Using the GUI
- **Upload Custom Image**: Click on the "Upload Custom Image" button to upload an image for logo detection. Custom images can also be obtained from the `custom_image` folder.
- **Exit**: Click on the "Exit" button to close the application.

The GUI will display the uploaded image along with various processed versions (resizing & padding, color correction, sharpening, and smoothing). It will also show the detected logo classification and RGB histograms.

## Training the Model
To train the logo detection model, run the `logodetection_training_testing.py` script:
```bash
python logodetection_training_testing.py --model_type <model_type>
```
Replace `<model_type>` with one of the following options: `lbp`, `resnet50`, or `resnet50_lbp`.

### Model Types

- **lbp**: This model uses Local Binary Patterns (LBP) for feature extraction. LBP is a simple and efficient texture operator that labels the pixels of an image by thresholding the neighborhood of each pixel and considers the result as a binary number. It is particularly useful for texture classification.

- **resnet50**: This model uses the ResNet-50 architecture, a deep residual network with 50 layers. ResNet-50 is a popular convolutional neural network architecture known for its effectiveness in image classification tasks. It includes skip connections or shortcuts to jump over some layers, making it easier to train deeper networks.

- **resnet50_lbp**: This model combines the features extracted using ResNet-50 with the Local Binary Patterns (LBP) features. This hybrid approach leverages both the deep learning capabilities of ResNet-50 and the texture classification power of LBP, potentially improving the accuracy of logo detection and classification.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The deep learning model is based on PyTorch and torchvision.
- The project uses Google Cloud Vision API for initial logo detection.

For further details, refer to the comments and documentation within each script.

---

Rename your file to `README.md` to ensure it is rendered correctly on GitHub.