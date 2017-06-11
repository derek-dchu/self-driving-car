# Term 1 Project 1 - Finding Lane Lines on the Road

The goals / steps of this project was to make a simple pipeline utilizing modeling blocks to detect lane lines on the road in a image. Same approach can be used to process videos frame by frame.

## outputs

![output1](processedSolidWhiteRight.gif?raw=true "solidWhiteRight")
![output2](processedSolidYellowLeft.gif?raw=true "solidYellowLeft")
![output3](extra.gif?raw=true "challenge")

---

### 1. Pipeline description.

My pipeline consisted of 7 steps:

1. resizing image
2. grayscale transformation
3. Gaussian smoothing
4. Canny edges detection
5. region of interest extraction
6. Hough lines extraction
7. lane lines extraction and drawing

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by adding following steps:

1. removing irrelevant line segments
2. dividing remaining line segments into two groups - left and right
3. finding the median of slope and intercept of both left and right groups

### 2. Shortcomings with current pipeline

We can eaisly observe some shortcomings with my curret pipeline in the chanllenge video.

1. extremely curved land lines.
2. complex road condition - objects shadow, different road segments cover with different materials.
3. fixed modeling parameters will only fit the video captured by a unique setup of carmer and car.


### 3. Suggest possible improvements to current pipeline

1. expand line detection into curve detection.
2. apply more advanced image transformation rather than use only grayscale. 
3. utilize machine learning techiques to compute a dymanic set of modeling parameters.