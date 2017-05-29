# **Vehicle Detection and Tracking** 

### This is a write up on the vehicle detection and tracking project


---

### **Goals of the vehicle detection and tracking Project**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/TrainingDataDistribution.png "Training data distribution"
[image2]: ./output_images/car_not_car.png "Sample Car and Non car images"
[image3]: ./output_images/FeatureVisuvalization.png "Feature extraction visualiztion"
[image4]: ./output_images/sliding_windows.jpg "Sliding windows"
[image8]: ./output_images/HOG_example.jpg
[image9]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

---

### *Histogram of Oriented Gradients (HOG)*

**1. Explain how (and identify where in your code) you extracted HOG features from the training images.**

The code for this reading and visualizing training data can be found under the "Extract training data" section of the IPython notebook.  
The code for extracting features is contained in the function **_extract_features()_** that can be found under the "Extract features for training" section of the IPython notebook.  

**_Data Preparation_**
* Car and non-car data was read from the training dataset provided
* The data was merged to create the full training dataset
* The number of car and non-car data is reviewed to ensure they are balanced.
* Sample car and non-car images are visualized to ensure the input data is not corrupted.

![alt text][image1]

![alt text][image2]

**_Feature Extraction Parameters_**

The final parameters used for feature extraction are shown below :

| Parameter                  |     Value	          | 
|:--------------------------:|:----------------------:| 
| Features considered	     | Color, Spatial, HOG	  |
| Color Space                | YCrCb				  |
| HOG orientations           | 9	 	              |
| HOG pixels per cell   	 | 8	 	              |
| HOG cells per block        | 2	 	              |
| HOG Channel       	     | All	 	              |
| Spatial binning dimensions | 32x32	 	          |
| Number of histogram bins   | 32	 				  |


**_Feature Extraction_**

* Each car and non car image is read and converted to YCrCb color space
* Spatial features are extracted using a 32 x 32 resize parameter across all 3 channels of the input image and stacked together
* Color histogram is extracted across all 3 channels of the input image and concatenated together
* HOG feature is extracted for all 3 channels and appended together
* The spatial, color and HOG features are appended together to produce the final image feature
* Total length of the final image feature is 8460
* The extracted feature is standarized using the sklearn.preprocessing.StandardScaler() function.


**_Feature Extraction Visualization_**

Below is the visualiztion of the feature extraction on a sample car and non image.

![alt text][image3]

**2. Explain how you settled on your final choice of HOG parameters.**

*The determination of the final parameters were more empherical.
*Various combinations were tried out and the combination that provided best accuracy consistently was chosen as the final parameter set.
*Training and prediction time was also a consideration when determining the parameters.  Parameters that provided marginal increase in accuracy with significant increase in prediction time were ignored.

### *3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).*

The code for training the classifier can be found under the "Train the classifier" section of the IPython notebook. 

* Training (80%) and Testing (20%) data are extracted from the standarized feature file after shuffling and ramdomizing.
* Linear SVM was used for training the classifier.
* An accuracy of 99% was achieved with a training time of ~ 8 seconds
* The model and feature extraction parameters are stored in a pickle file.  This will help us run the vehicle detection pipeline seperately at any point in time.

###Sliding Window Search

### *1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?*

The code for sliding window search is contained in the function **find_cars()_** that can be found under the "Vehicle Detection" section of the IPython notebook. 
The code for running the search and drawing boxes is contained in the function **detection_pipeline_** that can be found under the "Vehicle Detection Pipeline" section of the IPython notebook. 

*The search window is restricted to the bottom half of the image that has road and vehicles in it.  Top and bottom y limits of 400 and 656 has been applied for this project. 
*Four scales are used for this project - 1, 1.5, 2, 2.5.
*For each scale, the image is scanned for vehicles using sliding window search as described below : 
    * Input image is trimmed as per the y top and bottom limits provided. 
    * If a scale other than 1 is used, the image is resized proportionate to the scale. 
    * The block and step size are calculated using the resized image shape and cell_per_block parameters. 
    * The x and y steps are calculated using block size, blocks per window and cells per step. 
    * The image channel wise HOG features for the entire image are extracted ONLY ONCE per scale.  This will be reused when the sliding through the image instead of calculating every time for effeciency. 
    * A window is slide through the image to extract the color and spatial features.  The HOG features for the window is extracted from the already calculated full image HOG features. 
    * The color, spatial and HOG features are stacked together to form the final image feature. 
    * The final image feature is standardized using sklearn.preprocessing.StandardScaler() function. 
    * The classifier is run on the standardized feature to predict if the window has car image or not. 
    * If a car is detected in any of the windows during sliding, the box coordinates are calculated and appended to the list of detected boxes. 
    * The above process is repeated for each scale. 
*The result of the sliding window search is a list of boxes where cars were identified. 

The below image shows the full set of overlapping windows that will be used for detecting vehicles.    

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

*A heat threshold of 3 has been set for this project - only images that have atleast 3 detections are only considered as valid vehicles.  Others are filtered as false positive.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

