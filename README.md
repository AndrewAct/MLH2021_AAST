# MLH2021_AAST
# AAST- Andrew's Art Style Transfer

## Intro

Art Style Transfer is a hot topic in recent years, accompanied with the prosperity of AI. Generally, two images are needed. One is the content image, and one is the style reference image. The output images should be a blend of the two images.

A popular method is deploying CNN (Convolution Neural Network). And that's what the operator wants to use in this project.

## Methodology 

The operator wants to build a GUI to allow users to drop images. Then, this project will analyze the features and transfer the art style from one to another.

## Example 

### Content Image: 
![black-swan](https://user-images.githubusercontent.com/43218650/103468398-b4a8f600-4d26-11eb-8c9e-b10c79ca90fe.JPG)

### Style Image: 
![MonetLotus](https://user-images.githubusercontent.com/43218650/103468371-6a277980-4d26-11eb-9211-c4bb1c40a1bb.jpg)

### Blended: 
![BlendedSwan](https://user-images.githubusercontent.com/43218650/103468380-80353a00-4d26-11eb-8a4c-e2cb1c06a538.png)


## Note:
This project referred much to the TensorFlow open source project. The link is here: https://github.com/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb

The student also referred to this article on Medium: [Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution](https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398), by TensorFlow. 

Some bugs are not fixed completely, like how to cast uint8 into float without influencing loading images. I will keep trying to solve them.

Thank you for your time!
