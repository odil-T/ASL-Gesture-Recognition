# ASL-Gesture-Recognition

This repository contains the source code needed to run the ASL Gesture Recognition Streamlit app. The app can detect the ASL gestures and type the corresponding letters on a provided text box.

Visit the app through this link: [].

The `app_st.py` file stores the code used to launch the Streamlit app. This app uses model parameters stored in `models/model_NN_MP_for_st.h5` to make predictions.
The app can be run locally without Streamlit by running `app_local.py`. However, no key inputs were implemented for this file.

`model_NN_MP.py` was used to train and save a neural network model. A Kaggle dataset was used to provide image data of the various ASL gestures: []. Mediapipe was used to detect and save the hand landmarks. Those landmarks were then used as data to train the neural network model for classifying the gestures. The model was saved in three different formats in the `models` folder: `model_NN_MP_for_st.h5`, `model_NN_MP_for_st.keras`, and `model_NN_MP_for_st.tf`.
