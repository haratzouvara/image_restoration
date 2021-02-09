### Image restoration using the Pix2Pix [1] cGan and U-net [2] networks 

Pix2Pix network is used for image inpaiting.
Afterwards, back-to-back U-net networks are used for denoising and deblurring effect. 

#### References:

[1] https://arxiv.org/pdf/1611.07004.pdf

[2] https://arxiv.org/pdf/1505.04597.pdf


#### Examples:


<p float="left">
<img   src="images/noisy_140790.jpg"  hspace="20" width="150" >  
<img   src="images/gan_noisy_140790.jpg"  hspace="20" width="150">   
<img   src="images/denoising_deblurring_140790.jpg"  hspace="20" width="150" >  
<img   src="images/clean_140790.jpg"  hspace="20"  width="150">   
</p>

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) noisy_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) inpaiting_result_for(a) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (c) denoise_deblur_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (d) ground_truth
<p float="left">
<img   src="images/noisy_140776.jpg"  hspace="20" width="150" >  
<img   src="images/gan_noisy_140776.jpg"  hspace="20" width="150">   
<img   src="images/denoising_deblurring_140776.jpg"  hspace="20" width="150" >  
<img   src="images/clean_140776.jpg"  hspace="20"  width="150">   
</p>

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) noisy_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) inpaiting_result_for(a) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (c) denoise_deblur_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (d) ground_truth

<p float="left">
<img   src="images/image_gray_10.jpg"  hspace="20" width="150" >  
<img   src="images/gan_noisy_10.jpg"  hspace="20" width="150">   
<img   src="images/denoising_deblurring_10.jpg"  hspace="20" width="150" >  

</p>

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) noisy_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) inpaiting_result_for(a) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (c) denoise_deblur_image 


<p float="left">
<img   src="images/image_gray_16.jpg"  hspace="20" width="150" >  
<img   src="images/gan_noisy_16.jpg"  hspace="20" width="150">   
<img   src="images/denoising_deblurring_16.jpg"  hspace="20" width="150" >  
 
</p>

##### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (a) noisy_image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (b) inpaiting_result_for(a) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (c) denoise_deblur_image



### Requirements 
```
Python (suggested 3.7.1)  
Numpy   
os-sys  
OpenCv  
Tensorflow (suggested 2.4.0)  
