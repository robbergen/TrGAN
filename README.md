3-D PET Image Generation using TrGAN
====================================
This repository contains code used to generate synthetic 3-D Head and Neck PET images using Transversal GAN as seen in the following papers:

[3-D PET Image Generation with tumour masks using TGAN](https://arxiv.org/abs/2111.01866)

[Assessing Privacy Leakage in Synthetic 3-D PET Imaging using Transversal GAN](https://arxiv.org/abs/2206.06448)

Temporal Generative Adversarial Nets
====================================
This repository is based on TGAN:
[Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/abs/1611.06624).

## TGAN Requirements

These scripts require the following python libraries.

- Chainer 2.0.0+
- h5py
- numpy
- pandas
- PIL
- PyYAML
- matplotlib

Note that they also require ffmpeg to produce a video from a set of images.

## Usage

### Datasets
These scripts have been tested on the [HECKTOR 2020 dataset](https://www.aicrowd.com/challenges/miccai-2020-hecktor). 

In order to run the scripts, the dataset needs to be stored in .npy format in `/data`. The data should be 4-D (3 spatial dim and one channel dim) for unconditional generation and 5-D for conditional generation (image and mask are concatenated along 5th dimension).

We cropped our datasets centered on the bounding boxes included in the HECKTOR 2020 dataset. The code was only tested on sizes `64x64x64` and `64x64x32` due to memory constraints.

### Training

#### TrGAN with WGAN and Singular Value Clipping
There are a few configuration options in the original TGAN paper but only WGAN with SVD is tested for TrGAN PET image generation:
```
python train.py --config_path configs/hecktor/hecktor_wgan_svd_zdim-100_no-beta-all_init-uniform-all.yml --gpu 0
```

### Inference
``` 
python infer.py
```
Alternatively, see the jupyter notebook for visualization of results.


## Citation

Please cite these papers if you use our work:

```
@inproceedings{SPIE2022,
    author = {Bergen, R and Rajotte, J-F and Yousefirizi, F and Klyuzhin, IS and Rahmim, A and Ng RT},
    title = {3-D PET Image Generation with tumour masks using TGAN},
    booktitle = {SPIE},
    year = {2022},
}
```

## License

MIT License. Please see the LICENSE file for details.
