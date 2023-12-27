import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Read CSV file containing image paths and labels
csv_path = 'gender_detection.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_path)

# Define ImageDataGenerator for image augmentation and normalization
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Flow images from directory using ImageDataGenerator
train_validation_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory='E:\male&female identification',  # Replace with the root directory where 'train' folder is located
    x_col='file',
    y_col='gender',
    target_size=(128, 128),  # Adjust image size as needed
    batch_size=32,
    class_mode='binary',  # For binary classification (gender detection)
    subset='training',  # Using a subset of data for training
    shuffle=True  # Shuffle the data
)

# Constructing a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_validation_generator,
    epochs=10,  # Adjust number of epochs as needed
)
# After training
model.save('gender_detection_model.h5')
