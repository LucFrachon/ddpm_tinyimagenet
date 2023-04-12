# Denoising Diffusion Probabilistic Models, Trained on TinyImageNet

I implement the "Denoising diffusion probabilistic models" paper (Ho _et al._) [1] and train it on a TinyImageNet [2] class (n09428293, "seashore,
coast, seacoast, sea-coast"). Additionally, I scrape about 500 images from Google Image to roughly double the size of the training set. Even with only 1,000 images and limited data augmentation, the model is able to learn and produce decent pictures.

The validation set contains 50 images. The backbone is based on SmaAt-UNet [3] with added positional embedding layers. The model and training hyperparameters are optimised with WandB Sweps to minimise the Fréchet Inception Distance (FID) [4].

You can read the Medium stories based on the research I did to build this project: [Part1](https://medium.com/@luc.frachon/the-intuitive-diffusion-model-part-1-10155c69b944) and [Part 2](https://medium.com/@luc.frachon/the-intuitive-diffusion-model-part-2-79c7e1e0ecb1).

\[1\] Ho, J., Jain, A. and Abbeel, P., 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, pp.6840–6851.  
\[2\] Le, Y. and Yang, X., 2015. Tiny imagenet visual recognition challenge. CS 231N, 7(7), p.3.  
\[3\] Trebing, K., Staǹczyk, T. and Mehrkanoon, S., 2021. SmaAt-UNet: Precipitation nowcasting using a small attention-UNet architecture. 
Pattern Recognition Letters, 145, pp.178-186.
\[4\] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S., 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30.
