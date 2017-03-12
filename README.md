# Autonomous-Car-Project
This project aims at developing a Convolutional Neural Network which is efficient enough that it will drive the simulated car without crossing the lanes or leaving the road.This project aims at driving the car solely based upon the image inputs and nothing else.

# About the simulator
The simuator we are using is Udacity's Open Source Self Driving Car Simulator which is created in Unity 3D.It can be downloaded from https://github.com/udacity/self-driving-car-sim.

Detailed description and working of the simulator can be found at https://medium.com/towards-data-science/introduction-to-udacity-self-driving-car-simulator-4d78198d301d#.nbc8qcgf7.

The GitHub distribution includes the sample images. We include the following key files so you can get started quickly:

1. model.json, a json file storing the keras convolutional network
2. model.h5, weights after a dozen hours of training
3. drive.py, the autonomous server, run with "python drive.py model.json"

# Dependencies
 
1. numpy
2. flask-socketio
3. eventlet
4. h5py
5. pandas
6. keras
7. theano/tensorflow

# The Dataset
The dataset is entirely images of the track as seen by the center camera, left camera and right camera.I used only images from the central camera.
The dataset I collected can be found at https://drive.google.com/open?id=0B36796aguJo0SUw5ZUVUUkhEWlU.
I did some preprocessing like cropping the Region Of Interest (ROI).That code is available as image_preprocessing.py. However,the images which are there in the link are original ones and you will need to run the preprocessing code on those images.

# About the codes
The driving_log is a csv file which contains 6 columns.First three columns specify the name of the image (from left, center and right camera) saved while collecting the training data set.
The fourth column specifies the valuesof the steering associated with each image,while the fifth column specifies the acceleration and sixth column gives the speed of car.
NOTE: The values and names of images in driving_log.csv file are taken for the dataset which is shared on above google drive link.  

# Driving the car autonomously
Use the drive.py file to communicate between the simulator and the our CNN model. Also we need to a '.h5' file of the CNN model along with this drive.py file for complete autonomous driving using the CNN model we train. 

# Problems that are being faced
1. After 4 to 5 epochs the network starts overfitting. The test accuracy remains around 80%. Also this 80% accuracy on test data does not justify the results it gives.
2. The output I want to predict is the steering angle which lies between the values -1 to +1. The network is not able to predict the floating values lying between the above range of -1 to +1. 





