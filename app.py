import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Cache MobileNetV2 model loading
@st.cache_resource
def load_mobilenetv2():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

# Cache CIFAR-10 model loading
@st.cache_resource
def load_cifar10_model():
    return tf.keras.models.load_model('cifar10_model.h5')

def preprocess_image(image, target_size, normalize=False):
    """
    Preprocess the image by resizing, normalizing, and expanding dimensions.
    """
    image = image.convert("RGB")  # Convert to RGB if not already
    image = image.resize(target_size)
    img_array = np.array(image)
    if normalize:
        img_array = img_array.astype('float32') / 255.0  # Normalize for CIFAR-10
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        
        # Load MobileNetV2 model
        model = load_mobilenetv2()
        
        # Preprocess the image
        img_array = preprocess_image(image, target_size=(224, 224))
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for imagenet_id, label, score in decoded_predictions:
            st.write(f"**{label}**: {score * 100:.2f}%")

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        
        # Load CIFAR-10 model
        model = load_cifar10_model()
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img_array = preprocess_image(image, target_size=(32, 32), normalize=True)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.write(f"**Predicted Class**: {class_names[predicted_class]}")
        st.write(f"**Confidence**: {confidence * 100:.2f}%")

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("MobileNetV2 (ImageNet)", "CIFAR-10"))
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
