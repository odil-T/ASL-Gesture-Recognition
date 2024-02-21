# ASL-Gesture-Recognition

App link: []

This repository contains the source code needed to run the ASL Gesture Recognition Streamlit app. The app can detect the ASL gestures and type the corresponding letters on a provided text box.

### How to run the Streamlit app locally

1. Ensure you have python installed.
2. Download this repository.
3. Open the terminal in the local repository and run `pip install -r requirements.txt` to install the necessary libraries.
4. Enter `streamlit run app_st.py` in the terminal. The Streamlit app should open in a new window in your browser.
5. Close the terminal if you wish to close the app.

### Additional Information

`app_st_hosted.py` contains the code used to launch the Streamlit-hosted app. This app uses model parameters stored in `models/model_NN_MP_for_st.h5` to make predictions.

`app_st_local.py` contains the code that can launch the local Streamlit app. It uses the same model parameters as `app_st_hosted.py`.

`model_NN_MP.py` was used to train and save a neural network model. A Kaggle dataset was used to provide image data of the various ASL gestures: https://www.kaggle.com/datasets/grassknoted/asl-alphabet. Mediapipe was used to detect and save the hand landmarks from said images. Those landmarks were then used as data to train the neural network model for classifying the gestures. The model was saved in three different formats in the `models` folder: `model_NN_MP_for_st.h5`, `model_NN_MP_for_st.keras`, and `model_NN_MP_for_st.tf`.

The `sample_images` folder stores sample images of the different gestures for reference.

`model_CNN_canny.py` is an experimental file that was used to train and save another model. First, Canny edge detection was applied on the images from the dataset. Next, a convolutional neural network was trained and saved on the processed images.
