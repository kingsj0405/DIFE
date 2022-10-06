# Dense Interspecies Face Embedding (DIFE)
### [Project Page](https://yangspace.co.kr/dife/) | Video | Demo-Colab | Demo-StreamIt | Paper | Data | 
An official PyTorch implementation of the paper "Dense Interspecies Face Embedding".<br><br>
[Dense Interspecies Face Embedding](https://yangspace.co.kr/dife/)<br>
  [Sejong Yang](https://yangspace.co.kr)<sup>1</sup>,
  Subin Jeon<sup>1</sup>,
  [Seonghyeon Nam](https://shnnam.github.io/)<sup>2</sup>,
  [Seon Joo Kim](https://sites.google.com/site/seonjookim/)<sup>1</sup> <br>
  <sup>1</sup>Yonsei University, <sup>2</sup>York University <br>
in NeruIPS 2022

![image](https://user-images.githubusercontent.com/13496612/192178762-66e28752-de5e-4707-9634-a310ced9f0ff.png)

## Features
- [x] Inference for single image
- [ ] Training
- [ ] Experiment

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

### Demo

```
# python download.py --domain human+dog
# python inference.py --image_path demo/human_000001.png
# python inference.py --image_path demo/dog_000001.png
```

## Citation
```
@inproceedings{yang2022dife,
  title={Dense Interspecies Face Embedding},
  author={Sejong Yang and Subin Jeon and Seonghyeon Nam and Seon Joo Kim},
  year={2020},
  booktitle={ECCV},
}
```
