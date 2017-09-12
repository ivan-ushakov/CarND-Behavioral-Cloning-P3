# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Model Architecture and Training Strategy

For this project I decided to use convolutional network from publication "End-to-End Deep Learning for Self-Driving Cars". It has following architecture

![Architecture](resources/nvidia-cnn-architecture.png)

The only difference is Normalization layer. I used Lambda layer with equation `color / 255.0 - 0.5` for normalization and Cropping2D layer to remove some top and bottom part of input image and avoid learning of unnecessary features like sky and car. All other layers are the same as publication has (see model.py lines 56-72).

Here is model summary:

Layer (type)                    | Output Shape         | Param #    | Connected to                     
--------------------------------|----------------------|------------|----------------------
lambda_1 (Lambda)               | (None, 160, 320, 3)  | 0          | lambda_input_1[0][0]             
cropping2d_1 (Cropping2D)       | (None, 90, 320, 3)   | 0          | lambda_1[0][0]                   
convolution2d_1 (Convolution2D) | (None, 43, 158, 24)  | 1824       | cropping2d_1[0][0]               
convolution2d_2 (Convolution2D) | (None, 20, 77, 36)   | 21636      | convolution2d_1[0][0]            
convolution2d_3 (Convolution2D) | (None, 8, 37, 48)    | 43248      | convolution2d_2[0][0]            
convolution2d_4 (Convolution2D) | (None, 6, 35, 64)    | 27712      | convolution2d_3[0][0]            
convolution2d_5 (Convolution2D) | (None, 4, 33, 64)    | 36928      | convolution2d_4[0][0]            
flatten_1 (Flatten)             | (None, 8448)         | 0          | convolution2d_5[0][0]            
dense_1 (Dense)                 | (None, 100)          | 844900     | flatten_1[0][0]                  
dense_2 (Dense)                 | (None, 50)           | 5050       | dense_1[0][0]                    
dense_3 (Dense)                 | (None, 10)           | 510        | dense_2[0][0]                    
dense_4 (Dense)                 | (None, 1)            | 11         | dense_3[0][0]                    

* Total params: 981,819
* Trainable params: 981,819
* Non-trainable params: 0

For training I used Adam optimizer to avoid manual learning rate tuning and MSE for loss function. 

All traning data was collected with provided simulator. I used following strategy:
* One lap in normal direction
* One lap in reverse direction
* All three cameras used
* Each image has left-to-right flipped pair with inverted angle (see model.py lines 80-91)

Here is how one shot from three cameras looks like (left, center, right):

![Left Camera](resources/left_1.jpg)
![Center Camera](resources/center_1.jpg)
![Right Camera](resources/right_1.jpg)

Traning data has following distribution:

![Distribution](resources/distribution.png)

This gave me 19026 images in total (9513 records in CVS file). I used generator to avoid loading all this images in memory at once (see model.py lines 15-36). For validation I used 20% of the data. I don't use any overfitting techniques because my model has low number of epoch and I control learning process.

5 epoch was used for traning and here is how traning process looks like:

![Traning](resources/training.png)

After that I tried my model (see model.h5) with simulator and result was fine (see video.mp4). There were some lane crossing but car returned back to road quickly.

## Model Visualization

I was courious what features network tries to learn from input images. I used special package for Keras to visualize attention of last network layers (see visualize.html). I found that in general network tries to learn lane lines on road and use them to predict angle value. But also some wrong features were learned and I think that input data requires more preprocessing then it has now.
