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

Elastic Distortation：
----------------
Elasticity change is the random standard deviation of every dimension (-1,1) of a pixel. `Gauss filter (0,sigma)` is used to filter the deviation matrix of each dimension. Finally, the magnification factor alpha is used to control the deviation range. Thus, A'(x + delta_x, y + delta_y) is obtained from A (x, y). The value of a 'is obtained by interpolation in the original image, and the value of a' serves as the value in the original a position. Generally speaking, the smaller the alpha, the bigger the sigma, the smaller the deviation, and the closer the original image is.

For simple explanation of Elasticity Filter, see blog at https://blog.csdn.net/nima1994/article/details/79776802

The effect of applying elastic distortion to the MNIST image is as follows:
<br><br>
![affine result](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/elastic_transformation.png)

Codes of method Affine and Elastic Distortation are got from internet,not programmed by myself.
<br><br>
The Contrast between Affine and Elastic Distortation:
-------------

![affine result](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/compare.png)  

The structure of CNN is same as it illustrated in paper, and I implemented it using Pytorch.

At the project I tried three methods of image preprocessing, such as:
1. `Affine`
2. `Elastic Dis..`
3. `Affine + Elastic`

#### And got the results after 400 epoch training:
<br><br>
![affine result](https://github.com/FrankXu0808/Thesis_Reappearance-BestPracticeOfCNNinMnist/raw/master/images/results.png)  
  
The acc of my model did not reach the acc of paper, the reeson may be:
1. Epoch time
2. `hyper parameter:` Weight Initialization method,choice of activation function,epoch etc
3. The method details of expanding data set
4. Gradient descent method(Adam,SGD)

About the acc of MNIST at recent, see https://paperswithcode.com/sota/image-classification-on-mnist


