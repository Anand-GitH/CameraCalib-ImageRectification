# ComputerVision

Rotation Matrix - Image is rotated by camera angles and even can be rotated at certain angles and with rotation matrix we can recover the original image as rotation matrix has special property as the det(R)!=0 so inverse always exist so we can recover the image after multiple rotations using the rotation matrix.

Camera Calibration - Camera has two important metrics - Intrinsic metrics and Extrinsic metrics and to convert image to 3D world coordinate and from 3D to image coodrindate the inrtrinsic parameters are very important it includes parameters like focal length along x-axis and y-axis and offset along x and y axis.
Here from the given image we extract the image coordinates - corner points which are important features of an image and then map it with world coordinates 
And using linear combinations we derive intrinsic parameters. Intrinsic parameters are important to recover and project world objects from image.

Eight Point Algorithm: With just 8 same points from two different images we can derive the fundamental matrix with which we can project the 3D coordinates and draw epipolar lines and so on. 



 
