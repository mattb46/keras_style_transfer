# Overview #
The goal of this project is to show how perform style transfer using models in the Keras framework. The advantage of using Keras is that it provides a variety of pre-trained neural nets and an framework to quickly build and train custom models. Here we import the ResNet50 model with pre trained weights from ImageNet, use the Keras backend to construct a cost function consisting of style and content costs, then use an Adam optimizer to minimize this function.

## Before You Start ##
Please make sure you've updated Tensorflow and Keras and are currently running the newest versions.

## Resources ##
I'll be expanding this ReadMe to contain a more detailed explanation of style transfer and the code in general. In the meantime I've listed some resources below that are very useful.

* Coursera - Deep Learning: https://www.coursera.org/specializations/deep-learning
* Style Transfer: http://openaccess.thecvf.com/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
* ResNet: https://arxiv.org/abs/1512.03385
* Adam Optimizer: https://arxiv.org/abs/1412.6980

## Files ##
* style_keras.py is the main script which performes the style transfer with a pre trained model built in to Keras, ResNet50 was used in this example
* nst_utils.py is some helper functions used to normalize and reshape the input images, add noise, and save the generated images generated by style_keras.py
* The input folder contains the style and content images, experiment with different images and weights (in style_keras.py) to see what you can create
  * content.jpg -> content image of the DNA Bridge in Gainesville, FL https://en.wikipedia.org/wiki/Helyx_Bridge
  * style.jpg -> style image of Starry Night https://en.wikipedia.org/wiki/The_Starry_Night
* The output folder contains the generated image from the code as is in the repository
* gen0.jpg is the content image after adding noise
