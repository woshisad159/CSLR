# CSLR
This repo holds codes of the paper: Continuous Sign Language Recognition via Temporal Super-Resolution Network. [[paper]](https://arxiv.org/pdf/2207.00928.pdf)

---
### Notice

- This project is implemented in Pytorch (1.11.0+cu113). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- This project runs in pycharm, so you need to install pycharm

- The SLR is the main function.
---
### Data Preparation

1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).

2. After finishing dataset download, extract it.

---
### Inference

We provide the pretrained models (Down-sampling factor 1) for inference,
But the result that the models (downsampling factor 4) needs to be retrained:
|  Backbone  |  Down-sampling factor 1  |  Down-sampling factor 4  |
|            | WER on Dev | WER on Test | WER on Dev | WER on Test |
|  --------  | ---------- | ----------- | ---------- | ----------- |
| CNN+BiLstm | 26.1%      | 26.7%       | 29.8%      | 30.4%       |
|     VAC    | 21.8%      | 22.8%       | 25.1%      | 26.2%       |
|   MSTNet   | 20.3%      | 21.4%       | 23.4%      | 24.7%       |

---
### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@article{zhu2022continuous,
  title={Continuous Sign Language Recognition via Temporal Super-Resolution Network},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2207.00928},
  year={2022}
}
```

---
### Relevant paper

Multi-scale temporal network for continuous sign language recognition[[paper]](https://arxiv.org/pdf/2204.03864.pdf)

```latex
@article{zhu2022multi,
  title={Multi-scale temporal network for continuous sign language recognition},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2204.03864},
  year={2022}
}
```

---
### Acknowledge

- Thanks Yuecong Min, Aiming Hao et al for sharing the code \([`link`](https://github.com/ycmin95/VAC_CSLR)
