import tensorflow as tf


class ReadData():
    def __init__(self, train_path, test_path):
        self.shuffling = 500
        self.batch_size = 1
        self.img_width = 256
        self.img_height = 256
        self.train_path = train_path
        self.test_path = test_path

    def get_data(self):
        def load(image_file):
            image = tf.io.read_file(image_file)
            image = tf.image.decode_jpeg(image)

            image_div = (tf.shape(image)[1]) // 2

            noisy_image = image[:, :image_div, :]
            real_image = image[:, image_div:, :]

            noisy_image = tf.cast(noisy_image, dtype=tf.float32)
            real_image = tf.cast(real_image, dtype=tf.float32)

            return noisy_image, real_image

        def resize(noisy_image, real_image, height, width):
            noisy_image = tf.image.resize(noisy_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            return noisy_image, real_image

        def image_crop(noisy_image, real_image):
            stack_image = tf.stack([noisy_image, real_image], axis=0)
            crop_image = tf.image.random_crop(stack_image, size=[2, self.img_width, self.img_height, 3])

            return crop_image[0], crop_image[1]

        # normalizing the images to [-1, 1]
        def normalization(noisy_image, real_image):
            noisy_image = (noisy_image / 127.5) - 1
            real_image = (real_image / 127.5) - 1

            return noisy_image, real_image

        def data_train(image_file):
            noisy_image, real_image = load(image_file)
            noisy_image, real_image = image_crop(noisy_image, real_image)
            noisy_image, real_image = normalization(noisy_image, real_image)
            return noisy_image, real_image

        def data_test(image_file):
            noisy_image, real_image = load(image_file)
            noisy_image, real_image = resize(noisy_image, real_image, self.img_width, self.img_height)
            noisy_image, real_image = normalization(noisy_image, real_image)
            return noisy_image, real_image

        train_dataset = tf.data.Dataset.list_files(self.train_path + '/*.jpg')
        train_dataset = train_dataset.map(data_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.shuffling)
        train_dataset = train_dataset.batch(self.batch_size)

        test_dataset = tf.data.Dataset.list_files(self.test_path + '/*.jpg')
        test_dataset = test_dataset.map(data_test)
        test_dataset = test_dataset.batch(self.batch_size)

        return train_dataset, test_dataset

