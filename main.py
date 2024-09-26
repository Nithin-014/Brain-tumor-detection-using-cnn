# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import cv2

# # Function to preprocess the input image
# def preprocess_image(img, target_size=(150, 150)):
#     img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
#     img = cv2.resize(img, target_size)
#     img_array = np.array(img)
#     img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
#     img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
#     return img_array

# # Tensorflow Model Prediction
# def model_prediction(test_image):
#     model = tf.keras.models.load_model("braintumor.h5")
#     img_array = preprocess_image(test_image)
#     predictions = model.predict(img_array)
#     predicted_index = np.argmax(predictions)
    
#     labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
#     predicted_label = labels[predicted_index]
    
#     # Determine if it's a tumor or not
#     if predicted_label == 'no_tumor':
#         final_label = 'No Tumor'
#     else:
#         final_label = 'Tumor'
    
#     return final_label, predictions[0][predicted_index]

# # CSS for full background image
# page_bg_img = '''
# <style>
# .stApp {
#     background-image: url("th.jpg");
#     background-size: cover;
# }
# </style>
# '''

# # Apply the CSS
# st.markdown(page_bg_img, unsafe_allow_html=True)

# # Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# # Main Page
# if app_mode == "Home":
#     st.header("Brain Tumor Detection System")
#     st.write("This application uses a deep learning model to detect brain tumors from MRI images.")

# # About Project
# elif app_mode == "About Project":
#     st.header("About Project")
#     st.write("""
#     This project aims to develop a brain tumor detection system using deep learning techniques.
    
#     **Objectives**:
#     - Detect the presence of brain tumors from MRI images.
    
#     **Methodology**:
#     - Utilize a convolutional neural network (CNN) for image classification.
#     - Train the model on a dataset containing MRI images labeled with tumor types and non-tumor.
#     """)

# # Prediction Page
# elif app_mode == "Prediction":
#     st.header("Model Prediction")
#     test_image = st.file_uploader("Choose an MRI Image:")
#     if test_image is not None:
#         st.image(test_image, width=300, use_column_width=True)
    
#     # Predict button
#     if st.button("Predict"):
#         with st.spinner('Predicting...'):
#             predicted_label, confidence = model_prediction(test_image)
#             st.write("Prediction")
#             st.success(f"The model predicts: {predicted_label} with confidence {confidence:.2f}")










import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Function to preprocess the input image
def preprocess_image(img, target_size=(150, 150)):
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    img = cv2.resize(img, target_size)
    img_array = np.array(img)
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("braintumor.h5")
    img_array = preprocess_image(test_image)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    
    predicted_label = labels[predicted_index]
    
    # Determine if it's a tumor or not
    if predicted_label == 'no_tumor':
        final_label = 'No Tumor'
    else:
        final_label = 'Tumor'
    
    return final_label, predictions[0][predicted_index]

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("Brain Tumor Detection System")
    st.write("This application uses a deep learning model to detect brain tumors from MRI images.")
    image_path = "th.jpg"
    st.image(image_path, use_column_width=True)

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.write("""
    This project aims to develop a brain tumor detection system using deep learning techniques.
    
    **Objectives**:
    - Detect the presence of brain tumors from MRI images.
    
    **Methodology**:
    - Utilize a convolutional neural network (CNN) for image classification.
    - Train the model on a dataset containing MRI images labeled with tumor types and non-tumor.
    """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an MRI Image:")
    if test_image is not None:
        st.image(test_image, width=300, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            predicted_label, confidence = model_prediction(test_image)
            st.write("Prediction")
            st.success(f"The model predicts: {predicted_label} with confidence {confidence:.2f}")

