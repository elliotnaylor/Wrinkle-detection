# Wrinkle-detection
## Introduction
Wrinkle detection algorithm using facial landmark detection, gabor filtering, and segmentation. It is mentioned in more detail in the following publication:

 - Joint Roughness and Wrinkle Detection Using Gabor Filtering and Dynamic Line Tracking (https://www.cscjournals.org/library/manuscriptinfo.php?mc=IJCSS-1507)

## Description
A number of filters are applied in order of gray-scaling, gabor filtering, and frangi filering to find lines in the provided image, in this case wrinkles. This is followed by a novel vertical thresholding method to remove less dominant lines.
![image](https://user-images.githubusercontent.com/22525909/229803286-b95a78c0-4c20-434a-b77e-09405a7b9111.png)

This is followed by a novel line tracking algorithm to remove wrinkles that are below a certain length and to keep the most dominant. The images below show the first filtered image and after the thresholding, dot removal, and line tracking is applied.
![image](https://user-images.githubusercontent.com/22525909/229803397-c14edb5b-26d0-425a-b688-4bdaf20aa377.png)
