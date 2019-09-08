**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Before calibration"
[image2]: ./output_images/camera_cal_undistort_output/calibration1.jpg "Undistorted"
[image3]: ./test_images/test2.jpg "Test image"
[image4]: ./output_images/test_images_warped/test2.jpg "Warped"
[image5]: ./output_images/test_images_mag_binary/test2.jpg "Gradient-magnitude"
[image6]: ./output_images/test_images_l_channel_binary/test2.jpg "L-channel"
[image7]: ./output_images/test_images_s_channel_binary/test2.jpg "S-channel"
[image8]: ./output_images/test_images_final_binary/test2.jpg "Binary"
[image9]: ./output_images/test_images_sliding_windows/test2.jpg "Sliding windows"
[image10]: ./output_images/test_images_polynomial/test2.jpg "Lane polynomial"
[image11]: ./output_images/test_images_with_lanes/test2.jpg "Lane polynomial"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This refers to this README document.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd code cell of the IPython notebook located in "./Advanced-lane-finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

|Calibration image  | After distortion correction    | 
|:--------------------------------------------:|:--------:|
|![alt text][image1] | ![alt text][image2]|
  
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I used the calibration matrix values to undistort input images.
E.g.
![Undistorted image][]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.  

First of all, I warped the image to get a bird's eye view of the lane-image.
![alt text][image3]
![alt text][image4]

Then I used a combination of lightness, saturation and gradient-magnitude thresholds to generate a binary image. The final binray image (`combined`) is a combination of the various thresholds as follows:
```python
mag_binary = gradient_magnitude_mask(warped_img, blur=True, thresh_min=30, thresh_max=70)
    l_channel = hls_channel_mask(warped_img, 1, 100, 255)
    s_channel = hls_channel_mask(warped_img, 2, 100, 180)
    combined = np.zeros_like(s_channel)
    combined[((s_channel>0) & (l_channel>0)) | (mag_binary>0)] = 1
```
Here:  
a) `mag_binary` is the binary image with the sobel gradient in both x and y directions combined.  
![alt text][image5]  
b) `l_channel` is the Lightness channel mask of the image that was converted from RGB to HLS.  
![alt text][image6]  
c) `s_channel` is the Saturation channel mask of the image that was converted from RGB to HLS.  
![alt text][image7]  
d) The `combined` final image is a combination of points where saturation and lightness values are b/w the given thresholds or, the  gradient-magnitude is b/w its threshold.  
I came up with this strategy after observing that the saturation channel was consistently able to detect lane points very accurately, but had a few false positives (such as markings on th road or shadows). The lightness channel almost always detected the lanes, but also had a lot of false positives. So I did a bitwise `&` to find the common more accurate area. The gradient-magnitude channel gave much cripser and accurate lane boundaries than saturation, but suffered in cases where the lane lines got blurred in the distance. But overall, the gradient-magnitude improved the lane boundaries detection, which is why I kept it in the bitwise `|` condition with the rest.
![alt text][image8]  

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_birds_eye_view_lanes()`:  
```python
def get_birds_eye_view_lanes(img, src_points, dst_points):
    shape = img.shape
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    return warped_img
```
  The `get_birds_eye_view_lanes()` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points.  I calculated the source and destination points using a function called `transformation_points`, that takes an image (`img`) as input and uses the **trapezoidal** ROI that has the lanes (somewhere in the middle of the image to the bottom) as the source points. For the destination points, I used the corners of the original image.  
```python
def transformation_points(img):
    # Transform
    shape = img.shape
    xmax = shape[1]
    ymax = shape[0]
    offset = 90
    left_top= (xmax//2-offset, ymax//2+offset)
    right_top= (xmax//2+offset, ymax//2+offset)
    left_bottom= (0+offset, ymax-30)
    right_bottom= (xmax-offset,ymax-30)

    src_points = np.float32([left_top, right_top, right_bottom, left_bottom])
    dst_points = np.float32([(0, 0), (xmax, 0), (xmax, ymax), (0, ymax)])
    return src_points, dst_points
```


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 450      | 0, 0        | 
| 730, 450      | 1280, 0      |
| 1190, 690     | 1280, 720      |
| 90, 690       | 0, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here's an example of a transformed image:  

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


I used a sliding window technique to find the lanes in the image.

a) First, I used the final binary mask image (i.e., the `combined` numpy array) and created a histogram of its values using the lower 30% of the mask. I used only the lower 30% for the histogram as I only needed the starting window centers, whose maxima would be affected by the overall image.  
b) Using the histogram, I set the sliding window at the two maximas on the left and right. I calculated the mean position of the x-axis coordinates (i.e., along the width) of all the non-zero points within the two sliding windows  
c) I then proceeded to move the window up each time and calculated the mean position of the x-axis coordinates as usual. This gave me all the required x-axis points. Here's an image of the sliding windows along the mean x-axis points:
![alt text][image9]
d) Using the x-axis points for left and right lane from the sliding window, I fit a polynomial using the `fit_polynomial` function. The resulting image mask looks as follows:  
![alt text][image10]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature using the polynomial that I fit as mentioned above. The function to get the curvature uses the curvature formula for the point where the lane starts (i.e. bottom of the image-mask for the polynomial).

```python
def curvature(leftx, rightx, yaxis_points, left_fit, right_fit, ym_per_pix, xm_per_pix):
    left_fit_cr = np.polyfit(yaxis_points*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(yaxis_points*ym_per_pix, rightx*xm_per_pix, 2)
    y_eval = np.max(yaxis_points)

    left_curverad = np.float_power(1+np.square(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1]) ,1.5)/np.abs(2*left_fit_cr[0])  ## Implement the calculation of the left line here
    right_curverad = np.float_power(1+np.square(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1]) ,1.5)/np.abs(2*right_fit_cr[0])  ## Implement the calculation of the right line here
    return left_curverad, right_curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I created an image mask of the warped lane image with the fitted polynomial line and filled it with green color. 
```python
def draw_lane(poly_img_mask, left_fitx, right_fitx):
    for pt in range(poly_img_mask.shape[0]):
        poly_img_mask[pt][left_fitx[pt]:right_fitx[pt]+1] = 255
        
    return poly_img_mask
```
I then unwarped the mask and super-imposed it over the original image using the `cv2.addWeighted` method as shown:
```python
def add_lanes_to_img(img, lane_mask):
    rgb_mask = np.dstack((np.zeros_like(lane_mask), lane_mask, np.zeros_like(lane_mask)))
    return cv2.addWeighted(np.array(img, dtype=np.uint8),1.0, np.array(rgb_mask, dtype=np.uint8),0.5,0)
```
Here's the output of that:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### Issue #1: Unable to get good thresholds for magnitude and saturation level channels
I experimented with different levels of thresholds but couldn't seem, to find a general approach that works well for all the images.
Solution: I ended up warping the image to get the bird's eye perspective first so that I could better reason about my thresholding logic.

##### Issue #2: Lanes are not properly detected in shadows
I've been using HLS channels along with the gradient magnitude. I will next try using the HSV colorspace instead to see if I can get rid of noise created due to shadows. Currently this is why the lane detection is mediocre on the challenge video, and useless on the harder challenge video.
