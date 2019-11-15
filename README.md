Best Practices for Convolutional Neural Networks
Applied to Visual Document Analysis(2003)
=======
Introduction
---------
1. Show that neural networks achieve the best performance on a handwriting recognition task (MNIST)
2. The optimal performance on MNIST was achieved using two essential practices. 
First, we created a new,general set of elastic distortions that vastly expanded the size of the training set. 
Second, we used convolutional neural networks.


`Affine:`
translations, rotations.....

`Elastic Distortation:`
distortion Image Using Gauss Distribution


Affine:
----------
![affine method](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/affine.png)  

### Applying affine transformation to MNIST images:<br><br>
![affine result](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/affine_exa.png)  

Bilinear interpolation:
--------------
A simple algorithm for evaluating the grey level is “bilinear interpolation” of the pixel values of the original image.Although other interpolation schemes can be used (e.g.,bicubic  and  spline  interpolation),  the  bilinear interpolation `is one of the simplest and works well for generating additional warped characters image at the chosen resolution `.
<br><br>
### An example of Bilinear interpolation:
![affine method](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/Bilinear_interpolation.png) 
