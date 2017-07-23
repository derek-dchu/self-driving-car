# Term 1 Prject 5 - Vehicle Detection

The goals of this project is implementing a vehicle detection pipeline to detect and box vehicle from video.

The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_vs_non_car]: ./output_images/car_vs_non_car.png
[car_hog_Y]: ./output_images/car_hog_Y.png
[car_hog_Cr]: ./output_images/car_hog_Cr.png
[car_hog_Cb]: ./output_images/car_hog_Cb.png
[non_car_hog_Y]: ./output_images/non_car_hog_Y.png
[non_car_hog_Cr]: ./output_images/non_car_hog_Cr.png
[non_car_hog_Cb]: ./output_images/non_car_hog_Cb.png
[result]: ./output_images/detection.png
[frame1]: ./output_images/frame1.png
[frame2]: ./output_images/frame2.png
[frame3]: ./output_images/frame3.png
[frame4]: ./output_images/frame4.png
[frame5]: ./output_images/frame5.png
[frame6]: ./output_images/frame6.png
[frame1_labels]: ./output_images/frame1_labels.png
[frame2_labels]: ./output_images/frame2_labels.png
[frame3_labels]: ./output_images/frame3_labels.png
[frame4_labels]: ./output_images/frame4_labels.png
[frame5_labels]: ./output_images/frame5_labels.png
[frame6_labels]: ./output_images/frame6_labels.png
[frame6_result]: ./output_images/frame6_result.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the Section 2.2 of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car vs non-car][car_vs_non_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are example outputs of each channel using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![car_hog_Y][car_hog_Y]
![car_hog_Cr][car_hog_Cr]
![car_hog_Cb][car_hog_Cb]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out that result of parameter set, `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, has relatively clear separation between car images and non car images and the total of number features is not that big which is good for preventing overfitting.

If we compare above with following example of non car image, we can see that YCrCb color space extracts clear important car patterns such as shape, lights.

![non_car_hog_Y][non_car_hog_Y]
![non_car_hog_Cr][non_car_hog_Cr]
![non_car_hog_Cb][non_car_hog_Cb]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 2 types of features.

##### HOG features
above

##### Color Features
Histogram of HSV color channels with `32` bins and Spatial bins of HSV color channels with size `32x32`.

By combining with above HOG features, I used total of 7900 images with has ~3900 car images and 3900 non-car images to train a linear SVM. Each image has 8460 features by combining color and HOG features above. To fine tune parameters, I used [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to search for the best parameters from a list of candidates.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since computing HOG features is time consuming. To increase performance of my detector, instead of going to sliding windows, I decide to use HOG-subsampling window searching. By thinking `n x n` cells of HOG as a window, we can sliding precomputed HOG features into windows and then compute other features within the window. In addition, I limited the search area to lower area of the image.

I picked the scales and overlap ratio by experimenting different combinations and find the one that generate a desired amoung of detection windows of a object (3 ~ 4 for cars, 0 ~ 1 for non-cars).

Implementation can be found in Section 4.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![result][result]
---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here are six frames and their corresponding heatmaps:

![frame1][frame1]
![frame2][frame2]
![frame3][frame3]
![frame4][frame4]
![frame5][frame5]
![frame6][frame6]

##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![frame1_labels][frame1_labels]
![frame2_labels][frame2_labels]
![frame3_labels][frame3_labels]
![frame4_labels][frame4_labels]
![frame5_labels][frame5_labels]
![frame6_labels][frame6_labels]

##### Here the resulting bounding boxes are drawn onto the last frame in the series:

![frame6_result][frame6_result]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Although the pipeline is able to bound box around cars, the time used to process is relatively too high. In order to handle video with 24 FPS, we need to complete processing of each image within around 45 ms. Some performance improvement technique such as HOG sub-sampling has been implemented, however, it is stil not meeting the requirement. Other improvements can be made such as reducing total number of sampling windows by considering distribution of cars by distance.

The current pipeline will likely fail to detect cars that are far away or very close to the carmer as I am limited number of scaled sampling windows beacuse of the high processing time.
