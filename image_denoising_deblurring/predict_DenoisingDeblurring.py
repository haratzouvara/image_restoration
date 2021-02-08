from tensorflow.keras.models import load_model
import cv2
import numpy as np
from matplotlib import pyplot as plt

model = load_model('deblur_denoise.h5')

image = cv2.imread('paint_060000.jpg')
image = cv2.resize(image,(256,256))
plt.imshow(image, cmap='gray')
plt.show()


output = model.predict((image/255).reshape(-1,256,256,3))
output = np.asarray(output)

denoising_image = output[0,0].reshape(256,256)
plt.imshow(denoising_image, cmap= 'gray')
plt.show()
deblurring_image = output[1,0].reshape(256,256)
plt.imshow(deblurring_image, cmap='gray')
plt.show()