Here's the updated `README.txt` file for your GitHub repository:

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
└── README.txt                               # Provides an overview, setup instructions, and usage guidelines for the logo detection system project
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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The deep learning model is based on PyTorch and torchvision.
- The project uses Google Cloud Vision API for initial logo detection.

For further details, refer to the comments and documentation within each script.

---

Feel free to modify this `README.md` to suit your specific needs and provide any additional information that may be necessary.