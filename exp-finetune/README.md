
Experiment.

This directory does the following:

- Crops FGVC-Aircraft images in `preprocess.ipynb`
    - Output: images in `./dataset/cropped/*`
- Runs those images through Google's ViT transformer (`vit-base-patch16-224` on Hugging Face) in `process.ipynb`.
    - Output: `./dataset/embedding.py`
- Uses those embeddings to train a NN in `single.py`.
    - Output: ...
