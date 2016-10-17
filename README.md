# Brahms

Self Driving Car Simulator for Udacity [Challenge #2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.5650j9v4s)


This is to help evaluate your models. Given the model and images, steering angle, speed, time, the simulator will try to show how the car will move using the predicted steerings. 

##### Requirements
* OpenCV
* Scipy
* Numpy
* Pygame

### Usage

Check out `sim_runner.py` for example usage

[![Sample Video](http://img.youtube.com/vi/fAsRJ7-8Rb0/0.jpg)](https://www.youtube.com/watch?v=fAsRJ7-8Rb0 "Brahms")


### Motivation

At instance `t0`, the heading for both real and predicted car is the same (0 degree)
We get `real_steering` and `predicted_steering` from this step

At `t1`, we calculate the heading caused by each of the above steering angles (using `speed at t0` and `t1-t0`). The difference in these angles (`theta_error`) is the amount that we will rotate our image. That is, if the car was steered according to prediction instead of the real thing, the image visible to the camera will be rotated by this much.

At `t2`, we again calculate the above diff and add it to `theta_error`. This will be the new rotation

A failure scenario is reached when either

  * Rotation of the image is not possible
  * An upper limit of rotation angle is reached

**We could also have used net horizontal displacement for crashing out (like in the NVIDIA paper).**

**Also, I am not using horizontal displacement in the simulator as of now. May be in the next version I will use the left and right images and stich them together to get translation error too.**

### Questions and Contributions

Reach out to me (@yousuf) on Udacity's [Slack Channel](https://nd013.slack.com/messages/challenge-two/)


_Adapted from comma.ai's [view_steering_model.py](https://github.com/commaai/research/blob/master/view_steering_model.py)_
