import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot  as plt
from pix2pix_cGan import pix2pix

image = cv2.imread('10.png', 0)
plt.imshow(image, cmap='gray')
plt.show()

image = cv2.resize(image,(256,256))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
result, encimg = cv2.imencode('.jpg', image, encode_param)
image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)


image = image/127.5 - 1
cGan = pix2pix(checkpoint_dir=r'\training_checkpoints', batch_size=1, restore_checkpoints= True)

prediction = pix2pix.generate_images(cGan.generator, image.reshape(-1,256,256,3), image.reshape(-1,256,256,3))
feature_name = r'paint_060000.jpg'
prediction = np.squeeze(prediction)
max_prediction = np.max(prediction)
min_prediction = np.min(prediction)
final_image = (prediction- min_prediction) / (max_prediction - min_prediction)
matplotlib.image.imsave(feature_name, final_image, cmap='gray')