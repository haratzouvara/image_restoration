from Denoising_deblurring_ReadData import ReadData
from DenoisingDeblurringNet import DenoisngDeblurring
import os

path_noisy_train = r'F:\celeb_denoising_deblurring'
path_noisy_test = r'F:\celeb_denoising_deblurring'
batch_size = 16

load_dataset = ReadData(path_noisy_train, path_noisy_test, batch_size=batch_size, shuffling=500)
train_dataset, test_dataset= load_dataset.get_data()

list = os.listdir(path_noisy_train)
train_files = len(list)

list = os.listdir(path_noisy_test)
test_files = len(list)

steps = train_files // batch_size
val_steps = test_files // batch_size

if __name__ == '__main__':
   denoisingdeblurring = DenoisngDeblurring().encoder_decoder()
   denoisingdeblurring.compile(optimizer='adam', loss=['mse', 'mse'])
   denoisingdeblurring.fit(train_dataset, epochs=50, steps_per_epoch= steps, validation_data = test_dataset, validation_steps= val_steps)
   denoisingdeblurring.save('deblur_denoise.h5')
