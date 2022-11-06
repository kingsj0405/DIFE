# Dense Interspecies Face Embedding (DIFE)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-interspecies-face-embedding/interspecies-facial-keypoint-transfer-on-mafl)](https://paperswithcode.com/sota/interspecies-facial-keypoint-transfer-on-mafl?p=dense-interspecies-face-embedding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dense-interspecies-face-embedding/interspecies-facial-keypoint-transfer-on-wflw)](https://paperswithcode.com/sota/interspecies-facial-keypoint-transfer-on-wflw?p=dense-interspecies-face-embedding)
### [Project Page](https://yangspace.co.kr/dife/) | [Paper](https://openreview.net/forum?id=m67FNFdgLO9) | [Poster](https://yangspace.co.kr/dife/img/dife_poster.png) | Colab(Coming Soon)
An official PyTorch implementation of the paper "Dense Interspecies Face Embedding".<br><br>
[Dense Interspecies Face Embedding](https://yangspace.co.kr/dife/)<br>
  [Sejong Yang](https://yangspace.co.kr)<sup>1</sup>,
  Subin Jeon<sup>1</sup>,
  [Seonghyeon Nam](https://shnnam.github.io/)<sup>2</sup>,
  [Seon Joo Kim](https://sites.google.com/site/seonjookim/)<sup>1</sup> <br>
  <sup>1</sup>Yonsei University, <sup>2</sup>York University <br>
in NeurIPS 2022

![image](https://user-images.githubusercontent.com/13496612/192178762-66e28752-de5e-4707-9634-a310ced9f0ff.png)

## Features
- [x] Inference for single image
- [ ] Training
- [ ] Experiment
  - [ ] Interspecies keypoint transfer
  - [ ] Interspecies face parsing
- [ ] Demo with streamlit

## Setup

### Docker

- Build image

```
$ cd docker
$ docker build -t dife .
$ cd ..
```

- Run docker container and access

```
$ docker run -ti -d --gpus=all --name=dife -v .:/workspace --ipc=host dife
$ docker exec -ti dife /bin/bash
# (Do something)
```

### Pre-trained model

```
# python download.py --key model-human+dog
```

### Test data

```
# python download.py --
```

## Usage

### Inference for single image
```
# python inference.py --image_path demo/human_000001.png
# python inference.py --image_path demo/dog_000001.png
```

![image](https://user-images.githubusercontent.com/13496612/194290085-3a5b4112-c805-4c5a-afa6-71cc557ff53b.png)

## Citation
```
@inproceedings{yang2022dife,
  title={Dense Interspecies Face Embedding},
  author={Sejong Yang and Subin Jeon and Seonghyeon Nam and Seon Joo Kim},
  year={2022},
  booktitle={NeurIPS},
}
```

## Reference
- [jamt9000/DVE](https://github.com/jamt9000/DVE)
- [facebookresearch/detectron2/projects/DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
- [warmspringwinds/segmentation_in_style](https://github.com/warmspringwinds/segmentation_in_style)
