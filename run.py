import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('gender_detection_model.h5')

# Function to preprocess an image before feeding it to the model
def preprocess_image_cv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = cv2.resize(img, (128, 128))  # Resize the image to match the input size of the model
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Open the default webcam (source 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    # Preprocess the frame for gender prediction
    preprocessed_frame = preprocess_image_cv(frame)

    # Perform prediction using the loaded model
    prediction = model.predict(preprocessed_frame)

    # Interpret the prediction result
    gender = "Female" if prediction[0][0] >= 0.5 else "Male"

    # Display gender prediction on the frame
    cv2.putText(frame, f"Gender: {gender}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Gender Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
