# Face Detection and Classification

This project aims to detect faces in an image and classify the detected faces into emotion categories and gender using deep learning models. The application is built using Streamlit, a Python library for creating web applications.

## Features
- Upload an image containing faces for detection and classification
- Detect faces in the uploaded image using YOLO face detection model
- Classify each detected face into emotion categories using an emotion classification model
- Classify each detected face into gender categories using a gender classification model
- Display the results including the detected faces, predicted emotions, and predicted genders in a table format

## Requirements
- Python 3.x
- Streamlit
- PyTorch
- TensorFlow
- UltraLytics (for YOLO face detection)
- Other dependencies specified in `requirements.txt`

## Installation
1. Clone this repository:

    ```
    git clone https://github.com/your-username/face-detection-classification.git
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

## Usage
1. Navigate to the project directory:

    ```
    cd face-detection-classification
    ```

2. Run the Streamlit application:

    ```
    streamlit run main.py
    ```

3. Upload an image containing faces for detection and classification.

## Models
- YOLOv8-face: Pre-trained YOLOv8 model for face detection.
- Emotion Classification Model: Pre-trained deep learning model for classifying emotions (e.g., Happy, Sad, Angry, etc.).
- Gender Classification Model: Pre-trained deep learning model for classifying gender (Male or Female).

## Credits
- YOLOv8-face: [https://github.com/akanametov/yolov8-face](https://github.com/akanametov/yolov8-face)
- UltraLytics: [https://github.com/ultralytics](https://github.com/ultralytics)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
