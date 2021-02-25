from imgaug import augmenters as iaa
import numpy as np
import cv2

images = cv2.imread('00000323_4.PNG')
images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

#images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)


seq = iaa.Sequential([iaa.Fliplr(0.5),
                      iaa.GaussianBlur((0, 3.0))])

img_aug = seq.augment_images(images)

cv2.imshow('before', images)
cv2.imshow('aaa', img_aug)
cv2.waitKey(0)
cv2.destroyAllWindows()