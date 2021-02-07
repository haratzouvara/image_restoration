import random
import numpy as np
import cv2
import os
import matplotlib

folder = r'F:\img_align_celeba\img_align_celeba'


masks_number = 90

#define mask threshold range
low_thresh = 100
high_thresh = 150

#define coding quality coding range
low_cod = 70
high_cod = 95

#define blur window range
low_blur = 1
high_blur = 3

#define variance and sigma range
low_var = 1
high_var = 30

low_sigma = 0.3
high_sigma = 0.5

#define brightness range of deterioration
low_type = 1
high_type = 5

types_thresh = np.zeros((5,2))
types_thresh[0:5,0] = (200, 160, 110, 70, 20)
types_thresh[0:5,1] = (255, 210, 165, 115, 75)




for name in os.listdir(folder):
        #define image path
        path_ = os.path.join(folder, name)


        #read face image
        face = cv2.imread(path_, 0)
        face = cv2.resize(face, (256, 256))

        # select random mask
        ran_mask = random.randint(1, masks_number)
        path_mask = 'masks\%d.jpg' % ran_mask

        #read mask image
        mask_ = cv2.imread(path_mask, 0)
        mask_ = cv2.resize(mask_, (256,256))

        #random thresholding mask images
        rand_thresh = random.randint(low_thresh, high_thresh)
        ret, bw_mask = cv2.threshold(mask_, rand_thresh, 255, cv2.THRESH_BINARY)


        #change image coding
        #define random coding quality
        encoding_rand = random.randint(low_cod, high_cod)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), encoding_rand]
        result, encimg = cv2.imencode('.jpg', face, encode_param)
        encoding_image = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)

        #change bluring
        #define random blurring window
        rand_blur = random.randint(low_blur, high_blur)
        blur_image = cv2.blur(face, (rand_blur, rand_blur))

        #salt and pepper
        mean = 0
        var = random.randint(low_var, high_var)
        sigma = var * random.uniform(low_sigma, high_sigma)
        gaussian = np.random.normal(mean, sigma, (256,256))
        added_image = blur_image + gaussian


        #define mask intensity
        final_image = 255 * np.ones((256,256))
        type = random.randint(low_type, high_type)

        #final image
        for i in range(0, 255):
            for j in range(0, 255):
                if bw_mask[i, j] == 0:
                    final_image[i, j] = added_image[i, j]
                else:
                    if type == 1:
                        value = random.randint(types_thresh[0,0], types_thresh[0,1])
                        final_image[i, j] = value
                    elif type == 2:
                        value = random.randint(types_thresh[1,0], types_thresh[1,1])
                        final_image[i, j] = value
                    elif type == 3:
                        value = random.randint(types_thresh[2,0], types_thresh[2,1])
                        final_image[i, j] = value
                    elif type == 4:
                        value = random.randint(types_thresh[3,0], types_thresh[3,1])
                        final_image[i, j] = value
                    elif type == 5:
                        value = random.randint(types_thresh[4,0], types_thresh[4,1])
                        final_image[i, j] = value


        #     plt.imshow(final_image,cmap='gray')
        #     plt.show()

        #save images
    #    feature_name = 'F:\celeb_gray_inpaiting_noisy_blur\%s' % name
    #    matplotlib.image.imsave(feature_name, final_image, cmap='gray')

    #    feature_name = 'F:\celeb_gray_with_inpaiting\%s' % name
    #    matplotlib.image.imsave(feature_name, added_image, cmap='gray')

    #    feature_name = 'F:\celeb_gray_blur_denoising\%s' % name
    #    matplotlib.image.imsave(feature_name, blur_image, cmap='gray')

    #    feature_name = 'F:\celeb_gray_gt\%s' % name
    #    matplotlib.image.imsave(feature_name, face, cmap='gray')
