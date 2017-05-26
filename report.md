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

## Rubric Points

---
### Writeup / README

This project is a part of my udacity self driving car degree. The goal of the project is to learn state of the art image recognition for recognizing traffic signs. The dataset used was a dataset of german traffic singns from a competion at the IJCNN [1](#Houben2013). A summary of the data is provided in [Data Set Summary & Exploration](#dataset). The technique used for recognition is Convolutional Neural Networks, also referred as CNN or ConvNet. Experiments with different models of ConvNets were done to finally use a variation of the inception model with batch normalization having 96.2% on the validation set and XXXX on the test set. The code of the project can be found at [project code](https://github.com/joergsimon/SDCND-Term1-TrafficSign/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration<a name="dataset"></a>

The dataset used in the project is the german traffic sign detection benchmark, in short GTSDB used for a competion 2013 at the IJCNN [H](#Houben2013). The dataset was however already slightly modified for the project by udacity. Contrary to the original dataset the provided files by udacity were all already cropped to the sign in the center with the same resolution of 32x32 for each image. In total 51.839 images are in this dataset for the competion, of those I used 67.13% (34.799 samples) for training, 8.5% (4.410 samples) for validation and 24.36% (12.630 samples) for final testing. Many of the numbers of this report are numbers calculated from the validation set, as all teaking and experimentation is done with test and validation set to avoid test set contamination. Only directly before submission the final number on the test set is updated in the report.

The dataset contains 43 different german traffic signs, some of them blurred or occluded [1](#Houben2013). In the udacity version the data is already formated to have the sign in the center and a resolution of 32x32 pixel. Each image is a RGB color coded image, so the shape is (32,32,3) for one image in the dataset. Two example images are shown below. The whole data set is a piceled python numpy array with all the images inside.

A random image is visualised from the dataset:
![alt text][image1]

Each image has 3 color channels. For later normalization it is interesting to see in what values each channel ranges. A combined histogram showing the number of occurances of a color value on y and the color value (ranging from 0 to 255) on x over the whole dataset shows that there is interestingly only little variation overall in the color channels in over the whole dataset. It also shows that the value distribution is strongly skewed. There is a small peak in lower range of color saturation and a big peak in the high values.

The similarity of the overall color distribution together with the fact that earlier papers found that greyscale still performs better than color images [1](#Sermanet2011) lead to the motivation to convert the dataset to a greyscale dataset. Since the overall color distribution is already almost the same the greyscale transformation changes very little on the overal distribution of values, as can be seen in Figure X. The same random sample picture can be seen after the greyscale transformation in Figure X2.

All this analysis can be found in the cells under "Step 1: Dataset Summary & Exploration" in the [project code](https://github.com/joergsimon/SDCND-Term1-TrafficSign/Traffic_Sign_Classifier.ipynb).

### Design and Test a Model Architecture

#### Preprocessing

Only very little preprocessing was done for the images. Basically only conversion to greyscale. Since batch normalization layers are used in ever layer the moving the data to the center and normalize variance is taken care of anyway. Crafting features was not done because the CNN should learn this transformation by itself. What mainly would have been interesting is using data augmentation to have additional training samples and make the model less prone to things like rotation, shift or blur. Out of time reasons this was not done.

#### Model training

The model was trained with 67.13% (34.799 samples) for training, 8.5% (4.410 samples) for validation. 24.36% (12.630 samples) were kept for final testing at the moment this report is written. Since no data augmentation was used, there is no difference.

#### Final Model Architecture

The final architecture was based on the inception model. Ony one raw convolution layer was used at the start and without batch normalization (6 filters of 5x5). After that 6 inception layers follow with a final number of 512 filters in the last layer. After the first two and the fourth layer pooling is used to reduce the input dimension. The last inception is avaraged pooled over all 512 filters, and then three fully connected layers reduce to the final result. All activations are based on RELU activations as they are a simple model and still converging faster then f.e. sigmoid. ELU was not used as I did not know them untill more towards the end of the Term,

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

The training is in the same section of the notebook a little bit below. The Epoc size was set to 6 since after 6 the model usually fluctuaded with increased learning and actually decreased learning on the validation set, which is a usual sign of overfitting. Batch size of 128 seemd to be a reasonable default. As an optimizer the AdamOptimizer [3](#Kingma2015) is used, who uses momentum f.e. to overcome saddle points in the error function. The learning rate was choosen based on experiences in the class...

#### Approach

My final model results were:
* training set accuracy of ?
* validation set accuracy of 95,7 
* test set accuracy of ?

First I tried different general models and tune the parameter in the sense of tuning the filter number, sizes and similar. I also read a lot of papers on CNNs. Some are quotet, since I write that last minute, some quotes I might fill in later. Basically I could not tune the LeNet-5 Architecture over 93% at first for me, so I tried another architecutre. An already inception like architecture was references as baseline architecuture in the project, so I tried this one. Three paralell streams of convolution filters are computed, and in the end concatenated. The relatively direct translation of the architecture (as much as possible at least) resulted in a worse performance than the LeNet-5. With some adoptions it got better than LeNet-5 but still below 90%. I then read about the inception model, batch normalization, dropout and other techniques, and assembled a simple inception architecture, which yielded a good result on the validation set for me. For details on the architecture see above.

Basically most parameters were choosen because of values reported in papers, like 0.5 for dropout, or that I reduce the number of filters for the intermediate 1x1 convolutions in the inception. A high Dropout was nesecary because of overfitting. Also I think if I would have taken the complete inception v2 architecture but without pretrained weights the model would likely have overfittet. In the end the architecture is some empirically grown network of several small results.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

## References:

[1]<a name="Houben2013"></a> Houben et al. (2013), Sebastian Houben, Johannes Stallkamp, Jan Salmen, Marc Schlipsing and Christian Igel, Detection of Traffic Signs in Real-World Images: The {G}erman {T}raffic {S}ign {D}etection {B}enchmark, The 2013 International Joint Conference on Neural Networks (IJCNN)

[2]<a name="Sermanet2011"></a> Pierre Sermanet and Yann LeCun 2011, Traffic sign recognition with multi-scale Convolutional Networks, The 2011 International Joint Conference on Neural Networks (IJCNN)

[3]<a name="Kingma2015"></a> Diederik P. Kingma and Jimmy Lei Ba 2015, ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION, 3rd International Conference for Learning Representations, San Diego, 2015 (ICLR)
