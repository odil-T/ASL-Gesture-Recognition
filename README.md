# ASL-Gesture-Recognition

App link: []

This repository contains the source code needed to run the ASL Gesture Recognition Streamlit app. The app can detect the ASL gestures and type the corresponding letters on a provided text box.

### How to run the Streamlit app locally

1. Install Python - Please make sure you have python 3.x installed.
2. Clone The Repository - Open a terminal/command prompt and navigate to the directory where you want to store the application. Then, run the following command to clone the repository: `git clone https://github.com/odil-T/ASL-Gesture-Recognition.git`
3. Make A New Environment - Navigate into the cloned repository directory with `cd ASL-Gesture-Recognition` and run `pip install virtualenv`. Make a new virtual environment by running `virtualenv asl_app`. Activate the environment in Windows by running `asl_app\Scripts\activate` or in Mac/Linux by running `source asl_app/bin/activate`.
5. Install Dependencies - While the virtual environment is active, install the required libraries by running `pip install -r requirements.txt`.
6. Run The App - Once all the dependencies are installed, you can launch the application by running `streamlit run app_st_local.py`. This launches the Streamlit app locally. A new window should appear in your browser.
7. You can close the app by closing the terminal/command prompt. If you wish to open the app again, open the terminal/command prompt in `ASL-Gesture-Recognition` directory and run `asl_app\Scripts\activate` followed by `streamlit run app_st_local.py`. 

### Additional Information

`app_st_hosted.py` contains the code used to launch the Streamlit-hosted app. This app uses model parameters stored in `models/model_NN_MP_for_st.h5` to make predictions.

`app_st_local.py` contains the code that can launch the local Streamlit app. It uses the same model parameters as `app_st_hosted.py`.

`model_NN_MP.py` was used to train and save a neural network model. A Kaggle dataset was used to provide image data of the various ASL gestures: https://www.kaggle.com/datasets/grassknoted/asl-alphabet. Mediapipe was used to detect and save the hand landmarks from said images. Those landmarks were then used as data to train the neural network model for classifying the gestures. The model was saved in three different formats in the `models` folder: `model_NN_MP_for_st.h5`, `model_NN_MP_for_st.keras`, and `model_NN_MP_for_st.tf`.

The `sample_images` folder stores sample images of the different gestures for reference.

`model_CNN_canny.py` is an experimental file that was used to train and save another model. First, Canny edge detection was applied on the images from the dataset. Next, a convolutional neural network was trained and saved on the processed images.
