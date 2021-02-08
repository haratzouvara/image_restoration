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
<img   src="images/clean_140790.jpg"  width="150">   
</p>


### Requirements 
```
Python (suggested 3.7.1)  
Numpy   
os-sys  
OpenCv  
Tensorflow (suggested 2.4.0)  