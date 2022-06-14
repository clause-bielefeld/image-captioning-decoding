# Diversity as a By-Product: Goal-oriented Language Generation Leads to Linguistic Variation

This is the code for our paper ["Diversity as a By-Product: Goal-oriented Language Generation Leads to Linguistic Variation" (SIGDIAL 2021)](https://aclanthology.org/2021.sigdial-1.43/).

Steps for running the code:

1. Generate vocab and train model, put in `data/model` (cf. `model` submodule; trained model & vocab are provided)
2. Generate clusters of target and distractor images, put in `data/image_clusters` (cf. directory `generate_clusters`; clusters are provided)
3. Decode the model using the different methods / parameters (cf. directory `run_decoding`; generated captions are provided)
  - For RSA, we rely on the code for this paper: [Pragmatically Informative Image Captioning with Character-Level Inference; Reuben Cohn-Gordon, Noah Goodman, Christopher Potts](https://aclanthology.org/N18-2070/); cf. `rsa` submodule.
  - The code for the remaining decoding methods can be found in decoding.py. Parts of the ES implementation are based on [this implementation](https://github.com/saiteja-talluri/Context-Aware-Image-Captioning); parts of the nucleus / top-k decoding implementation on [this gist](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317).
  - file paths have to be added to the bash files before running them.

4. postprocess the generated captions
  cf. directory `postprocess_generated`

5. evaluate methods and place the results in `data/results`
  - cf. directory `evaluation`

6. display the results using the notebooks in `show_results`

# Data

Model, image clusters and generated captions can be found here:
https://drive.google.com/drive/folders/12FE3nEj7BlZyQwh_HQ9a-zqpVD8M68AO?usp=sharing
