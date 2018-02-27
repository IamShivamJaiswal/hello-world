# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model has 5 layers with descriptions as:
![Model Architecture](model_architecture.png)
1. **Layer 1**: Conv layer with 32 5x5 filters, followed by ELU activation
2. **Layer 2**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
3. **Layer 3**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 512 neurons and ELU activation

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting in Layer 2(model.py lines 116), Layer 3(model.py lines 122), and Layer4(model.py lines 129). 

The model was trained and validated on data sets provided by Udacity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

Optimizer: Adam Optimizer

No. of epochs: 3

Images generated per epoch: 20,000 images generated on the fly

Validation Set: 3000 images, generated on the fly

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used data sets provided by Udacity for training my model for first track. For the second track I generated my own data sets by keeping car on the mid of lane.Beacuse in second track there's lot of  up and down in the roads , some sharp edge corner which is not similar to first track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the [NVIDEA paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because it was one of the fine architecture for traininig neural network.So I started with 3 Conv layer and 2 fully connected layer.I used augmentation technque [this blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.d779iwp28) from this post. In the augmentation, I choose randomly the camera to take the image from center, left, right
and then randomly flipping and adding random brightness for preparing data for test and validation.
I choose 3 epochs , 20,000 image per epoch to train my model based on above architecure.
After this , vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 
1. **Layer 1**: Conv layer with 32 5x5 filters, followed by ELU activation
2. **Layer 2**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
3. **Layer 3**: Conv layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 512 neurons and ELU activation.

```py
def get_model():
    model = Sequential()
    # model.add(Lambda(preprocess_batch, input_shape=(160, 320, 3), output_shape=(64, 64, 3)))

    # layer 1 output shape is 32x32x32
    model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # layer 2 output shape is 15x15x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))
    model.add(MaxPooling2D((2, 2), border_mode='valid'))

    # layer 3 output shape is 12x12x16
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(Dropout(.4))

    # Flatten the output
    model.add(Flatten())

    # layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

    # layer 5
    model.add(Dense(512))
    model.add(ELU())

    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
