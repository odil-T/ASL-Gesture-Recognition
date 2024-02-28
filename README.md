# ASL-Gesture-Recognition

This repository contains the source code needed to run the ASL Gesture Recognition Streamlit app. The app can detect the ASL gestures and type the corresponding letters on a provided text box.

App sample images:
![app_screenshot_1](https://github.com/odil-T/ASL-Gesture-Recognition/assets/142138394/25806f94-1408-41f5-9430-d216bf273fe5)
![app_screenshot_2](https://github.com/odil-T/ASL-Gesture-Recognition/assets/142138394/f892683c-6438-44cf-8c93-addb717dba9a)
![app_screenshot_3](https://github.com/odil-T/ASL-Gesture-Recognition/assets/142138394/ca984d7a-7585-4588-bcd2-7ba29e2a468a)



### How to run the Streamlit app locally

1. Install Python - You need a python 3.x installation.
2. Clone The Repository - Open a terminal/command prompt and navigate to the directory where you want to store the application. Then, run `git clone https://github.com/odil-T/ASL-Gesture-Recognition.git` to clone the repository.
3. Make A New Environment - Navigate into the cloned repository directory with `cd ASL-Gesture-Recognition` and run `pip install virtualenv`. Run `virtualenv asl_app` to make a new virtual environment. Run `asl_app\Scripts\activate` in Windows or `source asl_app/bin/activate` in Mac/Linux to activate the environment.
5. Install Dependencies - While the virtual environment is active, install the required libraries by running `pip install -r requirements.txt`.
6. Run The App - Once all the dependencies are installed, you can launch the application by running `streamlit run app_st_local.py`. This launches the Streamlit app locally. A new window should appear in your browser.
7. You can close the app by closing the terminal/command prompt. If you wish to reopen it, open the terminal/command prompt in `ASL-Gesture-Recognition` directory and run `asl_app\Scripts\activate`, followed by `streamlit run app_st_local.py`. 

### Additional Information

`app_st_local.py` contains the code that can launch the local Streamlit app. This file uses model parameters stored in `models/model_NN_MP_for_st.h5` to make predictions.

`model_NN_MP.py` was used to train and save a neural network model. A Kaggle dataset was used to provide image data of the various ASL gestures: https://www.kaggle.com/datasets/grassknoted/asl-alphabet. Mediapipe was used to detect and save the hand landmarks from said images. Those landmarks were then used as data to train the neural network model for classifying the gestures. The model was saved in three different formats in the `models` folder: `model_NN_MP_for_st.h5`, `model_NN_MP_for_st.keras`, and `model_NN_MP_for_st.tf`.

The `sample_images` folder stores sample images of the different gestures for reference.

`model_CNN_canny.py` is an experimental file that was used to train and save another model. First, Canny edge detection was applied on the images from the dataset. Next, a convolutional neural network was trained and saved on the processed images.
