
# Image Augmentation in OpenPose

One single annotated image is corresponding to serveral training instances in OpenPose due to image augmentation, which depends on the number of annoations (person) in the image.

## 1. Original

An image with 3 persons (3 annotations).

![](img000000175734._01.jpg)

## 2. Scale image according to a bbox height

transformed_bbox.height = random_multiplier * 368 * 0.6

![](img000000175734._02.jpg)

## 3. Rotate image

random rotate from -40 degrees to 40 degrees

![](img000000175734._03.jpg)

## 4. Chrop image at the bbox center

resulted size = 368 * 368

![](img000000175734._04.jpg)
