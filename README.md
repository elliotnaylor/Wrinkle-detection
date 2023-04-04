# Wrinkle-detection
This technique was developed for a company looking to detect unhealthy and aged skin using a number of computer vision algorithms and for the recommendeation of creams. The algorithms have been optimised for mobile devices.

## Introduction
Wrinkle detection algoithm using a number of filters including gabor, frangi and a novel thresholding method. This is followed by a novel line tracking algorithm to detect the biggest wrinkles. The algorithm was tested on the AGES dataset with an accuracy of 79.41%. It is mentioned in more detail in the following publication:

 - Joint Roughness and Wrinkle Detection Using Gabor Filtering and Dynamic Line Tracking (https://www.cscjournals.org/library/manuscriptinfo.php?mc=IJCSS-1507)

## Description
A number of filters are applied in order of gray-scaling, gabor filtering, and frangi filering to find lines in the provided image, in this case wrinkles. The images below show the image before and after using the edge detection filters.

![image](https://user-images.githubusercontent.com/22525909/229820059-a1e838f2-73cf-4466-81ae-3ad87b3f1650.png)

![image](https://user-images.githubusercontent.com/22525909/229819963-279af1a6-50f4-4e7e-bcdf-ce7bba831d31.png)

After this process a novel vertical thresholding and blob removal algorithms are applied to the technique to remove any horizontal wrinkles and any areas below a certain size.

![image](https://user-images.githubusercontent.com/22525909/229820723-c1359f54-24c0-4fb5-97cd-99e355990041.png)

This is followed by a novel line tracking algorithm to remove wrinkles that are below a certain length and to keep the most dominant. The images below show the first filtered image and after the thresholding, dot removal, and line tracking is applied.

![image](https://user-images.githubusercontent.com/22525909/229821132-5762e2f0-679c-4212-b29c-2f785aa201d6.png)
