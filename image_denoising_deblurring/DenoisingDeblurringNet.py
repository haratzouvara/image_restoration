import tensorflow as tf

class DenoisngDeblurring():

    def conv_block(self, **params):
        filters = params["filters"]
        kernel_size = params["kernel_size"]
        strides = params.setdefault("strides", (1, 1))
        padding = params.setdefault("padding", "same")
        kernel_initializer = params.setdefault("kernel_initializer", "he_normal")

        def conv_block(input):
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                       kernel_initializer=kernel_initializer)(input)
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
            x = tf.keras.activations.relu(x)
            return x

        return conv_block

    def encoder_decoder(self):

        inputs = tf.keras.layers.Input((None, None, 3))
        conv1 = self.conv_block(filters=16, kernel_size=(1, 1))(inputs)
        conv1 = self.conv_block(filters=16, kernel_size=(1, 1))(conv1)
        pool1 = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(conv1)

        conv2 = self.conv_block(filters=32, kernel_size=(3, 3))(pool1)
        conv2 = self.conv_block(filters=32, kernel_size=(3, 3))(conv2)
        pool2 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv2)

        conv3 = self.conv_block(filters=64, kernel_size=(3, 3))(pool2)
        conv3 = self.conv_block(filters=64, kernel_size=(3, 3))(conv3)
        pool3 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv3)

        conv4 = self.conv_block(filters=128, kernel_size=(3, 3))(pool3)
        conv4 = self.conv_block(filters=128, kernel_size=(3, 3))(conv4)
        pool4 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv4)

        conv5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same")(pool4)
        conv5 = tf.keras.activations.relu(conv5)
        conv5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same")(conv5)
        conv5 = tf.keras.activations.relu(conv5)

        up1 = tf.keras.layers.UpSampling2D((2, 2))(conv5)
        up1 = self.conv_block(filters=128, kernel_size=(3, 3))(up1)
        merge1 = tf.keras.layers.concatenate([conv4, up1], axis=3)
        merge1 = self.conv_block(filters=128, kernel_size=(3, 3))(merge1)
        merge1 = self.conv_block(filters=128, kernel_size=(3, 3))(merge1)

        up2 = tf.keras.layers.UpSampling2D((2, 2))(merge1)
        up2 = self.conv_block(filters=64, kernel_size=(3, 3))(up2)
        merge2 = tf.keras.layers.concatenate([conv3, up2], axis=3)
        merge2 = self.conv_block(filters=64, kernel_size=(3, 3))(merge2)
        merge2 = self.conv_block(filters=64, kernel_size=(3, 3))(merge2)

        up3 = tf.keras.layers.UpSampling2D((2, 2))(merge2)
        up3 = self.conv_block(filters=32, kernel_size=(3, 3))(up3)
        merge3 = tf.keras.layers.concatenate([conv2, up3], axis=3)
        merge3 = self.conv_block(filters=32, kernel_size=(3, 3))(merge3)
        merge3 = self.conv_block(filters=32, kernel_size=(3, 3))(merge3)

        up4 = tf.keras.layers.UpSampling2D((2, 2))(merge3)
        up4 = self.conv_block(filters=16, kernel_size=(3, 3))(up4)
        merge4 = tf.keras.layers.concatenate([conv1, up4], axis=3)
        merge4 = self.conv_block(filters=16, kernel_size=(3, 3))(merge4)
        merge4 = self.conv_block(filters=16, kernel_size=(3, 3))(merge4)

        denoising = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))(merge4)
        denoising = tf.keras.activations.sigmoid(denoising)

        concat = tf.keras.layers.concatenate([inputs, denoising])

        conv1 = self.conv_block(filters=16, kernel_size=(1, 1))(concat)
        conv1 = self.conv_block(filters=16, kernel_size=(1, 1))(conv1)
        pool1 = tf.keras.layers.AveragePooling2D((2, 2), padding="same")(conv1)

        conv2 = self.conv_block(filters=32, kernel_size=(3, 3))(pool1)
        conv2 = self.conv_block(filters=32, kernel_size=(3, 3))(conv2)
        pool2 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv2)

        conv3 = self.conv_block(filters=64, kernel_size=(3, 3))(pool2)
        conv3 = self.conv_block(filters=64, kernel_size=(3, 3))(conv3)
        pool3 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv3)

        conv4 = self.conv_block(filters=128, kernel_size=(3, 3))(pool3)
        conv4 = self.conv_block(filters=128, kernel_size=(3, 3))(conv4)
        pool4 = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(conv4)

        conv5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same")(pool4)
        conv5 = tf.keras.activations.relu(conv5)
        conv5 = tf.keras.layers.Conv2D(256, (3, 3), kernel_initializer='he_normal', padding="same")(conv5)
        conv5 = tf.keras.activations.relu(conv5)

        up1 = tf.keras.layers.UpSampling2D((2, 2))(conv5)
        up1 = self.conv_block(filters=128, kernel_size=(3, 3))(up1)
        merge1 = tf.keras.layers.concatenate([conv4, up1], axis=3)
        merge1 = self.conv_block(filters=128, kernel_size=(3, 3))(merge1)
        merge1 = self.conv_block(filters=128, kernel_size=(3, 3))(merge1)

        up2 = tf.keras.layers.UpSampling2D((2, 2))(merge1)
        up2 = self.conv_block(filters=64, kernel_size=(3, 3))(up2)
        merge2 = tf.keras.layers.concatenate([conv3, up2], axis=3)
        merge2 = self.conv_block(filters=64, kernel_size=(3, 3))(merge2)
        merge2 = self.conv_block(filters=64, kernel_size=(3, 3))(merge2)

        up3 = tf.keras.layers.UpSampling2D((2, 2))(merge2)
        up3 = self.conv_block(filters=32, kernel_size=(3, 3))(up3)
        merge3 = tf.keras.layers.concatenate([conv2, up3], axis=3)
        merge3 = self.conv_block(filters=32, kernel_size=(3, 3))(merge3)
        merge3 = self.conv_block(filters=32, kernel_size=(3, 3))(merge3)

        up4 = tf.keras.layers.UpSampling2D((2, 2))(merge3)
        up4 = self.conv_block(filters=16, kernel_size=(3, 3))(up4)
        merge4 = tf.keras.layers.concatenate([conv1, up4], axis=3)
        merge4 = self.conv_block(filters=16, kernel_size=(3, 3))(merge4)
        merge4 = self.conv_block(filters=16, kernel_size=(3, 3))(merge4)

        deblurring = tf.keras.layers.Conv2D(filters= 1, kernel_size=(1, 1))(merge4)
        deblurring = tf.keras.activations.sigmoid(deblurring)

        model_c = tf.keras.Model(inputs=inputs, outputs=[denoising, deblurring])

        return model_c
