# Denoising Diffusion Probabilistic Models, trained on TinyImageNet

This is a personal project where I implemented a model inspired by the DDPM paper [1] and trained it on TinyImageNet [2], a subset of ImageNet
where the images are 64x64 pixels in size and with only 200 classes (500 images in each class). I only retained one class (n09428293, "seashore,
coast, seacoast, sea-coast") and scraped about 500 additional images from Google Image to roughly double the size of the training set. These images are provided 
in the repo. Perhaps surprisingly, even with only about 1,000 images, the model is able to learn and produce decent pictures, even with fairly limited data augmentation.

The validation set contains 50 images.

The backbone is based on SmaAt-UNet \[3\] with added positional embedding layers.

The 'outputs' folder contains examples of what the model is able to generate. The images are clearly not photorealistic, but definitely contain features consistent with 
seaside landscapes.

You can read the Medium post based on the research I did to build this project: [Part1](https://medium.com/@luc.frachon/the-intuitive-diffusion-model-part-1-10155c69b944) and [Part 2](https://medium.com/@luc.frachon/the-intuitive-diffusion-model-part-2-79c7e1e0ecb1).

\[1\] Ho, J., Jain, A. and Abbeel, P., 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, pp.6840–6851.  
\[2\] Le, Y. and Yang, X., 2015. Tiny imagenet visual recognition challenge. CS 231N, 7(7), p.3.  
\[3\] Trebing, K., Staǹczyk, T. and Mehrkanoon, S., 2021. SmaAt-UNet: Precipitation nowcasting using a small attention-UNet architecture. 
Pattern Recognition Letters, 145, pp.178-186.
