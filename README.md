# Online-Diffusion-Policy

Rough idea that I want to prototype using Video World Models 

First training the value of the model with simulator data (mix of working and false data) -> progressively train the world model + value --> pivot to training action policy.  Using a mix of actual model inference failues live and correct samples. 

I decided that joint training is better, but the ratio needs to be controlled and progressively allocated during training, a lot of data mixture that needs to be controlled, but i think i can just make an actor that monitors progress during training to do this for me. 


Decoupled Isaac Sim Environments -> Inference of Trainable World Model (weights reloaded periodically with new checkpoints) + correct data -> Ray Data -> Training Video World Model -> update world model in inference pipeline. 

