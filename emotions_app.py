import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import pickle

# Cache the loading of the model and training history
@st.cache_resource
def load_model_and_history():
    model = tf.keras.models.load_model('emotions_classifier.h5')
    with open('history.pkl', 'rb') as file:
        history = pickle.load(file)
    return model, history

# Load the saved model and training history
model, history = load_model_and_history()

# Define class labels for predictions
class_labels = ['angry', 'disgust', 'happy', 'love', 'nervous', 'sad', 'surprise']

# Function to make individual predictions
def predict_image(image):
    img = image.resize((128, 128))  # Resize image to 128x128
    img = img_to_array(img) / 255.0  # Convert image to array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return class_labels[np.argmax(prediction)]

# Streamlit app
st.title('GM Analytics Emotion Image Classifier')
st.write("Enter the URL of any Emotion image and the model will predict the emotion of the image. The Model includes prediction result, plot the evaluation chart of the model on a test dataset, and plot the training history.")
st.write("Example link: [Click and copy this Example Image link](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSrAPtLT44fxO96YvpN56Tgd8lfwtzQlg2AYg&s)")

# Input for the image URL
@st.cache_data
def get_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

image_url = st.text_input("Paste the URL of the image here:")

if image_url:
    try:
        image = get_image_from_url(image_url)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict_image(image)
        st.write(f'The person in the image is {label}.')
    except Exception as e:
        st.write(f"Error: {e}")

# Evaluate the model on the test data and display the results
# Assuming you have a directory called 'test' with subdirectories for each emotion
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'test',  # Update this to your test dataset path
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
st.write(f'Test Loss: {test_loss}')
st.write(f'Test Accuracy: {test_accuracy}')

# Plotting the history of the training
st.write("### Training History")

# Plot training & validation loss values
fig, ax = plt.subplots()
ax.plot(history['loss'], label='Training Loss')
ax.plot(history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
st.pyplot(fig)

# Plot training & validation accuracy values
fig, ax = plt.subplots()
ax.plot(history['accuracy'], label='Training Accuracy')
ax.plot(history['val_accuracy'], label='Validation Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracy')
ax.legend()
st.pyplot(fig)

# Function to plot predictions
def plot_predictions(model, generator, num_images=5):
    x_test, y_test = next(generator)
    y_pred = model.predict(x_test)

    fig, axes = plt.subplots(1, num_images, figsize=(20, 20))
    axes = axes.flatten()
    for img, true_label, pred_label, ax in zip(x_test, y_test, y_pred, axes):
        ax.imshow(img)
        ax.axis('off')
        true_class = class_labels[np.argmax(true_label)]
        pred_class = class_labels[np.argmax(pred_label)]
        ax.set_title(f"True: {true_class}\nPredicted: {pred_class}")
    plt.tight_layout()
    st.pyplot(fig)

# Plot predictions for the first batch of test images
if st.button("Show Predictions on Test Data"):
    plot_predictions(model, test_generator, num_images=5)

# streamlit run "C:/Users/User/Documents/NorthCentralUniversity/ModelDeployment/Emotions/emotions_app.py"
