# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road

I modified 

### 1. My pipeline consisted of 6 steps.

1 - Convert to grayscale
2 - Apply Gaussian smoothing
3 - Apply Canny
4 - Create Mask
5 - Apply Hough transform
6 - Manipulate Hough lines 


# Pipeline implementation:
def process_image(img):
    
    #Convert image to grayscale:
    gray = grayscale(img)
    
    #Apply gaussian smoothing:
    gaus = gaussian_blur(gray, Gauss_kernel)
    
    #Detect edges:
    edges = canny(gaus, canny_low, canny_high)  
    
    #Get shape of image 
    imshape = image.shape
    
    #Get vertices for mask:
    vertices = np.array([[(0,imshape[0]),(x_offset1, y_offset1), (x_offset2, y_offset1), (imshape[1],imshape[0])]], dtype=np.int32)   
    
    #Create mask:
    masked = region_of_interest(edges, vertices)
    
    #Apply Hough transformation:
    line_img = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
    
    #Manipulating Hough lines
    result = weighted_img(line_img, img)
    
    return result

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first filtering left and right lines.In Next step just take avg of all points for left lines and find slope of all avg points.Now, using avg slope and top filter calculate x_cordinates for top point.similarly calculate for bottom x_cordinates. Repeat the same for right lines.

2. Identify potential shortcomings with your current pipeline
One potential shortcoming would be  when there's sharpe angle of change it'll unable to take those lines in consideration

3. Suggest possible improvements to your pipeline
draw_lines further can modfied for optional challenge.
