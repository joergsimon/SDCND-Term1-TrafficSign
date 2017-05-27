# **Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/example_image.png "random sign from the dataset"
[image2]: ./images/total_color_distribution.png "total color distribution"
[image3]: ./images/example_image.png "random sign after greyscaling from the dataset"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./images/5imgs.png "5 new images"

## Rubric Points

---
### Writeup / README

This project is a part of my udacity self driving car degree. The goal of the project is to learn state of the art image recognition for recognizing traffic signs. The dataset used was a dataset of german traffic singns from a competion at the IJCNN [[1](#Houben2013)]. A summary of the data is provided in [Data Set Summary & Exploration](#dataset). The technique used for recognition is Convolutional Neural Networks, also referred as CNN or ConvNet. Experiments with different models of ConvNets were done to finally use a variation of the inception [[6](#Szegedy2015)] model with batch normalization [[4](#Ioffe2015)] having 96.2% on the validation set and XXXX on the test set. The code of the project can be found at [project code](https://github.com/joergsimon/SDCND-Term1-TrafficSign/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration<a name="dataset"></a>

The dataset used in the project is the german traffic sign detection benchmark, in short GTSDB used for a competion 2013 at the IJCNN [H](#Houben2013). The dataset was however already slightly modified for the project by udacity. Contrary to the original dataset the provided files by udacity were all already cropped to the sign in the center with the same resolution of 32x32 for each image. In total 51.839 images are in this dataset for the competion, of those I used 67.13% (34.799 samples) for training, 8.5% (4.410 samples) for validation and 24.36% (12.630 samples) for final testing. Many of the numbers of this report are numbers calculated from the validation set, as all teaking and experimentation is done with test and validation set to avoid test set contamination. Only directly before submission the final number on the test set is updated in the report.

The dataset contains 43 different german traffic signs, some of them blurred or occluded [[1](#Houben2013)]. In the udacity version the data is already formated to have the sign in the center and a resolution of 32x32 pixel. Each image is a RGB color coded image, so the shape is (32,32,3) for one image in the dataset. Two example images are shown below. The whole data set is a piceled python numpy array with all the images inside.

A random image is visualised from the dataset:
![alt text][image1]

Each image has 3 color channels. For later normalization it is interesting to see in what values each channel ranges. A combined histogram showing the number of occurances of a color value on y and the color value (ranging from 0 to 255) on x over the whole dataset shows that there is interestingly only little variation overall in the color channels in over the whole dataset. It also shows that the value distribution is strongly skewed. There is a small peak in lower range of color saturation and a big peak in the high values.

The similarity of the overall color distribution together with the fact that earlier papers found that greyscale still performs better than color images [[1](#Sermanet2011)] lead to the motivation to convert the dataset to a greyscale dataset. Since the overall color distribution is already almost the same the greyscale transformation changes very little on the overal distribution of values, as can be seen in Figure X. The same random sample picture can be seen after the greyscale transformation in Figure X2.

All this analysis can be found in the cells under "Step 1: Dataset Summary & Exploration" in the [project code](https://github.com/joergsimon/SDCND-Term1-TrafficSign/Traffic_Sign_Classifier.ipynb).

### Design and Test a Model Architecture

#### Preprocessing

Only very little preprocessing was done for the images. Basically only conversion to greyscale. Since batch normalization [[4](#Ioffe2015)] layers are used in ever layer the moving the data to the center and normalize variance is taken care of anyway. Crafting features was not done because the CNN should learn this transformation by itself. What mainly would have been interesting is using data augmentation to have additional training samples and make the model less prone to things like rotation, shift or blur. Out of time reasons this was not done.

#### Model training

The model was trained with 67.13% (34.799 samples) for training, 8.5% (4.410 samples) for validation. 24.36% (12.630 samples) were kept for final testing at the moment this report is written. Since no data augmentation was used, there is no difference.

#### Final Model Architecture

The final architecture was based on the inception model [[6](#Szegedy2015)]. Ony one raw convolution layer was used at the start and without batch normalization [[4](#Ioffe2015)]] (6 filters of 5x5). After that 6 inception layers follow with a final number of 512 filters in the last layer. After the first two and the fourth layer pooling is used to reduce the input dimension. The last inception is avaraged pooled over all 512 filters, and then three fully connected layers reduce to the final result. All activations are based on RELU activations as they are a simple model and still converging faster then f.e. sigmoid. ELU was not used as I did not know them untill more towards the end of the Term,

The code for my final model is located in Model Architecture -> Inception Model in the notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5, RELU | 1x1 stride, valid padding, outputs 28x28x6 	|
| INCEPTION 1			| outputs 28x28x16								|
| INCEPTION 2			| outputs 28x28x32 								|
| Max Pool 				| 2x2, outputs 14x14x32 						|
| INCEPTION 3			| outputs 14x14x64								|
| INCEPTION 4			| outputs 14x14x128								|
| Max Pool 				| 2x2, outputs 7x7x32 							|
| INCEPTION 5			| outputs 7x7x256								|
| INCEPTION 6			| outputs 7x7x512								|
| Avarage Pool			| outputs 1x1x512, aka. 512						|
| Fully Connected, RELU	| outputs 128. 									|
| Fully Connected, RELU	| outputs 84. 									|
| Dropout				| Propability of 0.5 for each cell 				|
| Fully Connected		| outputs 43. 									|

Each Inception layer is made the following way:

| INCEPTION Layer  		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1x1 to output    		| output: 1/4 of desired output   				| 
| 1x1 to 3x3    		| output: 1/8 of desired output   				| 
| 3x3 to output   		| output: 1/4 of desired output   				|
| 1x1 to 5x5    		| output: 1/8 of desired output   				| 
| 5x5 to output   		| output: 1/4 of desired output   				|
| Max Pool 3x3 stride 	| output: 1/4 of desired output   				| 

F.e. if I use an inception to have 64 filters then:

| INCEPTION Layer  		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1x1 to output    		| output: 16 Filters   							| 
| 1x1 to 3x3    		| output: 8 Filters fed to 3x3   				| 
| 3x3 to output   		| output: 16 Filters   							|
| 1x1 to 5x5    		| output: 8 Filters fed to 5x5   				| 
| 5x5 to output   		| output: 16 Filters   							|
| Max Pool 3x3 stride 	| output: 16 Filters   							| 

The training is in the same section of the notebook a little bit below. The Epoc size was set to 6 since after 6 the model usually fluctuaded with increased learning and actually decreased learning on the validation set, which is a usual sign of overfitting. Batch size of 128 seemd to be a reasonable default. As an optimizer the AdamOptimizer [[3](#Kingma2015)] is used, who uses momentum f.e. to overcome saddle points in the error function. The learning rate was choosen based on experiences in the class...

#### Approach

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 94.2% 
* test set accuracy of 93.7%

First I tried different general models and tune the parameter in the sense of tuning the filter number, sizes and similar. I also read a lot of papers on CNNs. Some are quotet, since I write that last minute, some quotes I might fill in later. Basically I could not tune the LeNet-5 Architecture over 93% at first for me, so I tried another architecutre. An already inception like architecture was references as baseline architecuture in the project, so I tried this one. Three paralell streams of convolution filters are computed, and in the end concatenated. The relatively direct translation of the architecture (as much as possible at least) resulted in a worse performance than the LeNet-5. With some adoptions it got better than LeNet-5 but still below 90%. I then read about the inception model [[6](#Szegedy2015)], batch normalization [[4](#Ioffe2015)], dropout [[5](#Srivastava2014)] and other techniques, and assembled a simple inception architecture, which yielded a good result on the validation set for me. For details on the architecture see above.

Basically most parameters were choosen because of values reported in papers, like 0.5 for dropout [[5](#Srivastava2014)], or that I reduce the number of filters for the intermediate 1x1 convolutions in the inception. A high Dropout [[5](#Srivastava2014)] was nesecary because of overfitting. Also I think if I would have taken the complete inception v2 architecture but without pretrained weights the model would likely have overfittet. In the end the architecture is some empirically grown network of several small results.

### Test a Model on New Images

Here are six German traffic signs that I found on the web:

![6 new images][image9]

The first and the second image have fractions of other images inside the image and are a bit occluded. The others should be relatively easy. The background is a bit different and so on. I also added a 7th image, a relatively easy 120km/h for testing, it funnyly confused the image with a 20km/h sign. For the Top 1 Klassification error the accuracy was 57.1%.

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution      	| General caution   							| 
| Children crossing    	| Right-of-way at the next intersection	        |
| No passing			| No passing									|
| Bicycles crossing	    | Speed limit (50km/h)			 				|
| Speed limit (70km/h)	| Speed limit (70km/h)      					|
| No vehicles	        | No vehicles     	                			|
| Speed limit (120km/h)	| Speed limit (120km/h)      					|

#### How that compares to the other results

The test with 7 new images yielded a accuracy of 57.1%. Training, validation and test was above 90% so one would expect a higher number here. The unseen test set had 12.630 items inside and still archieved a good accuracy which would mean we are not overfitting. However, it can easily be that there is some uniformity f.e. in the background the model is overfitting to which affects the whole dataset. To really find out if we were just really unlucky with the small sample size of really new images or are indeed overfitting an additonal test set of at least 100 images should be assembled.

#### Confidences

57.1% is not the highes rate but okayish. Generally the model was pretty certain of it's predictions. While the top 5 accuracy is way better (85.7%) if the label is in the top 5 its certainty is still most of the time almost zero.

The list of confidences per prediction is shown here:

###### General caution

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 0.998    				| 18    General caution 				    	|
| 0.002    				| 11    Right-of-way at the next intersection 	|
| 0.000    				| 27    Pedestrians 				        	|
| 0.000    				| 23    Slippery road 				        	|
| 0.000    				| 2    Speed limit (50km/h) 					|

###### Children crossing

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 0.894    				| 11    Right-of-way at the next intersection 	|
| 0.065    				| 29    Bicycles crossing 					    |
| 0.027    				| 21    Double curve 					        |
| 0.004    				| 24    Road narrows on the right 				|
| 0.003    				| 28    Children crossing 					    |

###### No passing

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 1.000    				| 9    No passing 					            |
| 0.000    				| 16    Vehicles over 3.5 metric tons prohibited    |
| 0.000    				| 1    Speed limit (30km/h) 					|
| 0.000    				| 10    No passing for vehicles over 3.5 metric tons |
| 0.000    				| 19    Dangerous curve to the left 			|

###### Bicycles crossing

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 0.446    				| 2    Speed limit (50km/h) 					|
| 0.239    				| 9    No passing 					            |
| 0.164    				| 1    Speed limit (30km/h) 					|
| 0.041    				| 3    Speed limit (60km/h) 					|
| 0.039    				| 5    Speed limit (80km/h) 					|

###### Speed limit (70km/h)

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 0.994    				| 4    Speed limit (70km/h) 					|
| 0.003    				| 0    Speed limit (20km/h) 					|
| 0.001    				| 8    Speed limit (120km/h) 					|
| 0.001    				| 7    Speed limit (100km/h) 					|
| 0.000    				| 3    Speed limit (60km/h) 					|

###### No vehicles

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 1.000    				| 15    No vehicles 					        |
| 0.000    				| 40    Roundabout mandatory 					|
| 0.000    				| 13    Yield 					                |
| 0.000    				| 1    Speed limit (30km/h) 					|
| 0.000    				| 38    Keep right 					            |

###### Speed limit (120km/h)

| Probability         	|     Prediction	     						|
|:---------------------:|:---------------------------------------------:| 
| 0.868    				| 0    Speed limit (20km/h) 					|
| 0.037    				| 1    Speed limit (30km/h) 					|
| 0.020    				| 7    Speed limit (100km/h) 					|
| 0.014    				| 8    Speed limit (120km/h) 					|
| 0.014    				| 14   Stop 					                |

As already writte above the top 5 accuracy was 85.7%

### Test a Model on Test Dataset

This was computed after the whole report is written. The score on the test data set is: 93.7%

## References:

[1]<a name="Houben2013"></a> Houben et al. (2013), Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing and Christian Igel, Detection of Traffic Signs in Real-World Images: The {G}erman {T}raffic {S}ign {D}etection {B}enchmark, The 2013 International Joint Conference on Neural Networks (IJCNN)

[2]<a name="Sermanet2011"></a> Pierre Sermanet and Yann LeCun 2011, Traffic sign recognition with multi-scale Convolutional Networks, The 2011 International Joint Conference on Neural Networks (IJCNN)

[3]<a name="Kingma2015"></a> Diederik P. Kingma and Jimmy Lei Ba 2015, ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, 3rd International Conference for Learning Representations, San Diego, 2015 (ICLR)

[4]<a name="Ioffe2015"></a> Sergey Ioffe and Christian Szegedy 2015, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

[5]<a name="Srivastava2014"></a> Srivastava et al. 2014, Dropout: a simple way to prevent neural networks from overfitting, The Journal of Machine Learning Research Vol 15 Issue 1

[6]<a name="Szegedy2015"></a> Szegedy et al. 2015, Going Deeper with Convolutions, in ILSVRC 2014 
