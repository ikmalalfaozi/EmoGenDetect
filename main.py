import streamlit as st
from PIL import Image
from ultralytics import YOLO
from keras.models import load_model
import numpy as np


# Function to load YOLO face detection model
def load_face_detection_model():
    model_path = './models/yolov8n-face.pt'
    model = YOLO(model_path)
    return model


# Function to load emotion classification model
def load_emotion_classification_model():
    model_path = './models/FER_ResNet.h5'  # Path to your emotion classification model
    model = load_model(model_path)
    return model


# Function to load gender classification model
def load_gender_classification_model():
    model_path = './models/GC_vgg16.h5'  # Path to your gender classification model
    model = load_model(model_path)
    return model


def detect_faces(image, face_detection_model):
    # detect face
    results = face_detection_model(image)

    # Retrieves all bounding boxes from the detection results
    boxes = results[0].boxes.data

    # Take the confidence score and bounding boxes that have confidence above 50%
    confidences = results[0].boxes.conf
    boxes = boxes[confidences > 0.5]

    return boxes


# Function to classify emotion from face image
def classify_emotion(face_image, emotion_model):
    # Preprocess the face image
    face_image = np.array(face_image.resize((48, 48)))
    face_image = np.expand_dims(face_image, axis=0)

    # Perform emotion classification
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_probabilities = emotion_model.predict(face_image)[0]
    predicted_emotion = emotion_labels[np.argmax(emotion_probabilities)]
    return predicted_emotion, emotion_probabilities


# Function to classify gender from face image
def classify_gender(face_image, gender_model):
    # Preprocess the face image
    face_image = np.array(face_image.resize((48, 48)))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)

    # Perform gender classification
    gender_labels = ['Female', 'Male']
    gender_probabilities = gender_model.predict(face_image)[0]
    predicted_gender = gender_labels[np.argmax(gender_probabilities)]
    return predicted_gender, gender_probabilities


# Main function
def main():
    # Load models
    face_detection_model = load_face_detection_model()
    emotion_model = load_emotion_classification_model()
    gender_model = load_gender_classification_model()

    # Streamlit UI
    st.title('Face Detection and Classification')
    uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect faces
        boxes = detect_faces(image, face_detection_model)
        results_table = []
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max, _, _ = box.tolist()
            face_image = image.crop((x_min, y_min, x_max, y_max))
            gray_image = face_image.convert('L')

            # Classify emotion
            predicted_emotion, emotion_probabilities = classify_emotion(gray_image, emotion_model)

            # Classify gender
            predicted_gender, gender_probabilities = classify_gender(gray_image, gender_model)

            # Add results to table
            results_table.append({
                'Face': i + 1,
                'Face Image': face_image,
                'Predicted Emotion': predicted_emotion,
                'Emotion Probability': round(emotion_probabilities.max(), 2),
                'Predicted Gender': predicted_gender,
                'Gender Probability': round(gender_probabilities.max(), 2)
            })

        # Display results in table
        col = st.columns(6)
        col[0].info('Number')
        col[1].info('Face Image')
        col[2].info('Predicted Emotion')
        col[3].info('Emotion Probability')
        col[4].info('Predicted Gender')
        col[5].info('Gender Probability')
        for result in results_table:
            col = st.columns(6)
            col[0].write(result['Face'])
            col[1].image(result['Face Image'], width=56)
            col[2].write(result['Predicted Emotion'])
            col[3].write(result['Emotion Probability'])
            col[4].write(result['Predicted Gender'])
            col[5].write(result['Gender Probability'])


if __name__ == '__main__':
    st.set_page_config(page_title="EmoGenDetect", page_icon="🤖", layout="wide")
    main()
