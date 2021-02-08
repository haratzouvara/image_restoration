import tensorflow as tf
import matplotlib.pyplot as plt

class ReadData():
    def __init__(self, train_path, test_path, batch_size, shuffling):
        self.shuffling = shuffling
        self.batch_size = batch_size
        self.img_width = 256
        self.img_height = 256
        self.train_path = train_path
        self.test_path = test_path



    def get_data(self):

        def load(image_file):
            image = tf.io.read_file(image_file)
            image = tf.image.decode_jpeg(image)

            image_div = (tf.shape(image)[1]) // 3

            noisy_blur_image = image[:, :image_div, :]
            denoising_image = image[:, image_div:2*image_div, :]
            real_image = image[:, 2*image_div:, :]


            noisy_blur_image = tf.cast(noisy_blur_image, dtype=tf.float32)
            denoising_image = tf.cast(denoising_image, dtype=tf.float32)
            real_image = tf.cast(real_image, dtype=tf.float32)

            return noisy_blur_image, denoising_image, real_image




        # normalizing the images to [0, 1]
        def normalization(noisy_blur_image, denoising_image, real_image):
            noisy_blur_image = noisy_blur_image / 255
            denoising_image = denoising_image / 255
            real_image = real_image / 255

            return noisy_blur_image, denoising_image, real_image

        def data_train(image_file):
            noisy_blur_image, denoising_image, real_image = load(image_file)
            noisy_blur_image, denoising_image, real_image = normalization(noisy_blur_image, denoising_image, real_image)
            return noisy_blur_image, (denoising_image, real_image)

        def data_test(image_file):
            noisy_blur_image, denoising_image, real_image = load(image_file)
            noisy_blur_image, denoising_image, real_image = normalization(noisy_blur_image, denoising_image, real_image)
            return noisy_blur_image, (denoising_image, real_image)

        train_dataset = tf.data.Dataset.list_files(self.train_path + '/*.jpg')
        train_dataset = train_dataset.map(data_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.shuffling)
        train_dataset = train_dataset.batch(self.batch_size).repeat()

        test_dataset = tf.data.Dataset.list_files(self.test_path + '/*.jpg')
        test_dataset = test_dataset.map(data_test)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset

