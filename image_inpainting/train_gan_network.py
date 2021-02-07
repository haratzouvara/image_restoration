from Inpainting_ReadData import ReadData
from pix2pix_cGan import pix2pix

load_dataset = ReadData(r'F:\output\train\celeb_inpaiting_multi_res', r'F:\output\val\celeb_inpaiting_multi_res' )
train_dataset, test_dataset = load_dataset.get_data()

if __name__ == '__main__':
    cGan = pix2pix(epochs= 40, checkpoint_dir= '/training_checkpoints', train_dataset= train_dataset, test_dataset= test_dataset,  batch_size= 1)
    cGan.train()