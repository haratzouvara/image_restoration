import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import datetime
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU


class pix2pix():

    def __init__(self, epochs=None, checkpoint_dir=None, train_dataset=None, test_dataset=None, batch_size=None, restore_checkpoints = False):
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.patch_shape = 8
        self.image_shape =  (256,256,3)





        #self.generate_images = generate_images

        self.discriminator = self.define_discriminator()
        self.disc_opt = Adam(lr=0.0002, beta_1=0.5)
        self.discriminator.compile(loss='binary_crossentropy', optimizer = self.disc_opt, loss_weights=[0.5])

        self.generator = self.define_generator()

        self.discriminator.trainable = False
        noisy = Input(self.image_shape)
        gen_output = self.generator(noisy)

        valid = self.discriminator([noisy, gen_output])

        self.gan = Model(inputs=[noisy], outputs=[valid, gen_output])
        self.gan_opt = Adam(lr=0.0002, beta_1=0.5)
        self.gan.compile(loss=['binary_crossentropy', 'mae'], loss_weights=[1, 100], optimizer=self.gan_opt)

        self.restore_checkpoints = restore_checkpoints

        checkpoint = tf.train.Checkpoint(generator_optimizer=self.gan_opt, discriminator_optimizer=self.disc_opt,
                                         generator=self.generator, discriminator=self.discriminator)

        if self.restore_checkpoints == True:
            checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

    @staticmethod
    def generate_images(gen, input, real):
        pred = gen(input, training=True)
        plt.figure(figsize=(15, 15))

        display_images = [input[0], real[0], pred[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(display_images[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
        return pred[0]



    def define_discriminator(self):

        init = RandomNormal(stddev=0.02)
        input_image = Input(shape=self.image_shape)
        real_image = Input(shape=self.image_shape)  # target, clean image
        concat = Concatenate()([input_image, real_image])  # concatenate source,target

        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(concat)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(patch_out)

        model = Model([input_image, real_image], patch_out)  # output

        return model

    def define_generator(self):

      init = RandomNormal(stddev=0.02)
      input = Input(shape=self.image_shape)

      en_conv1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input)
      en_conv1 = LeakyReLU(alpha=0.2)(en_conv1)

      en_conv2 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv1)
      en_conv2 = BatchNormalization()(en_conv2)
      en_conv2 = LeakyReLU(alpha=0.2)(en_conv2)

      en_conv3 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv2)
      en_conv3 = BatchNormalization()(en_conv3)
      en_conv3 = LeakyReLU(alpha=0.2)(en_conv3)

      en_conv4 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv3)
      en_conv4 = BatchNormalization()(en_conv4)
      en_conv4 = LeakyReLU(alpha=0.2)(en_conv4)

      en_conv5 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv4)
      en_conv5 = BatchNormalization()(en_conv5)
      en_conv5 = LeakyReLU(alpha=0.2)(en_conv5)

      en_conv6 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv5)
      en_conv6 = BatchNormalization()(en_conv6)
      en_conv6 = LeakyReLU(alpha=0.2)(en_conv6)

      en_conv7 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv6)
      en_conv7 = BatchNormalization()(en_conv7)
      en_conv7 = LeakyReLU(alpha=0.2)(en_conv7)

      b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(en_conv7)
      b = Activation('relu')(b)

      de_conv1 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(b)
      de_conv1 = BatchNormalization()(de_conv1)
      de_conv1 = Dropout(0.5)(de_conv1)
      de_conv1 = Concatenate()([de_conv1, en_conv7])
      de_conv1 = Activation('relu')(de_conv1)

      de_conv2 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv1)
      de_conv2 = BatchNormalization()(de_conv2)
      de_conv2 = Dropout(0.5)(de_conv2)
      de_conv2 = Concatenate()([de_conv2, en_conv6])
      de_conv2 = Activation('relu')(de_conv2)

      de_conv3 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv2)
      de_conv3 = BatchNormalization()(de_conv3)
      de_conv3 = Dropout(0.5)(de_conv3)
      de_conv3 = Concatenate()([de_conv3, en_conv5])
      de_conv3 = Activation('relu')(de_conv3)

      de_conv4 = Conv2DTranspose(512, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv3)
      de_conv4 = BatchNormalization()(de_conv4)
      de_conv4 = Concatenate()([de_conv4, en_conv4])
      de_conv4 = Activation('relu')(de_conv4)

      de_conv5 = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv4)
      de_conv5 = BatchNormalization()(de_conv5)
      de_conv5 = Concatenate()([de_conv5, en_conv3])
      de_conv5 = Activation('relu')(de_conv5)

      de_conv6 = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv5)
      de_conv6 = BatchNormalization()(de_conv6)
      de_conv6 = Concatenate()([de_conv6, en_conv2])
      de_conv6 = Activation('relu')(de_conv6)

      de_conv7 = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same',kernel_initializer=init)(de_conv6)
      de_conv7 = BatchNormalization()(de_conv7)
      de_conv7 = Concatenate()([de_conv7, en_conv1])
      de_conv7 = Activation('relu')(de_conv7)

      output = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(de_conv7)
      output = Activation('tanh')(output)

      model = Model(input, output)
      return model





    def train(self):

        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer= self.gan_opt, discriminator_optimizer=self.disc_opt,
                                         generator=self.generator, discriminator=self.discriminator)


        start_time = datetime.datetime.now()
        patch_shape = self.patch_shape
        batch_size = self.batch_size
        y_real = np.ones((batch_size, patch_shape, patch_shape, 1))
        y_fake = np.zeros((batch_size, patch_shape, patch_shape, 1))
        for epoch in range(self.epochs):

            for n, (train_input, train_real) in self.train_dataset.enumerate():

                gen_output = self.generator.predict(train_input)
                d_loss_real = self.discriminator.train_on_batch([train_input, train_real], y_real)
                d_loss_fake = self.discriminator.train_on_batch([train_input, gen_output], y_fake)
              #  d_loss_add = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.gan.train_on_batch([train_input], [y_real, train_real])
                elapsed_time = datetime.datetime.now() - start_time

                if n % 1000 == 0:
                    display.clear_output(wait= True)
                    for test_input, test_real in self.test_dataset.take(1):
                        pix2pix.generate_images(self.generator, test_input, test_real)
                        print(epoch, self.epochs, n, elapsed_time)

            # if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix= checkpoint_prefix)
