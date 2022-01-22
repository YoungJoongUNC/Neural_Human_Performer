# Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering
### [Project Page](https://youngjoongunc.github.io/nhp/) | [Video](https://www.youtube.com/watch?v=4b5SPwPOKVo) | [Paper](https://arxiv.org/pdf/2109.07448.pdf)


<img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/teaser.gif?raw=true" width="50%" height="50%" />

> [Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering](https://arxiv.org/pdf/2012.15838.pdf)  
> Youngjoong Kwon, Dahun Kim, Duygu Ceylan, Henry Fuchs  
> NeurIPS 2021 (Spotlight)

## Code and Instructions
Coming very soon !

## Installation

Please see [INSTALL.md](INSTALL.md).


## Run the code on ZJU-MoCap

Please see [INSTALL.md](INSTALL.md) to download the dataset.

### Visualization on ZJU-MoCap
1. Download the corresponding pretrained model and put it to `$ROOT/data/trained_model/if_nerf/demo/300.pth`.
2. Visualization:
    * Free-viewpoint rendering
    ```
    # render frames
    CUDA_VISIBLE_DEVICES=0 python run.py --type visualize --cfg_file configs/performance.yaml test_mode model_x_motion_x exp_name demo test.epoch 300 gpus "0,"
    
    # generate video
    python gen_video.py 
    ```
    <img src="https://github.com/YoungJoongUNC/Neural_Human_Performer/blob/main/image/subject_0_freeview.gif?raw=true" width="60%" height="60%" />

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{kwon2021neural,
  title={Neural human performer: Learning generalizable radiance fields for human performance rendering},
  author={Kwon, Youngjoong and Kim, Dahun and Ceylan, Duygu and Fuchs, Henry},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
## Acknowledgments

We thank [Sida Peng](https://pengsida.net/) of Zhejiang University, Hangzhou, China, for 
very many helpful discussions on a variety of implementation details of [Neural Body](https://zju3dv.github.io/neuralbody/).
We thank [Ruilong li](https://www.liruilong.cn/) and [Alex Yu](https://alexyu.net/) of UC Berkeley
for many discussions on the [AIST++](https://google.github.io/aichoreographer/) dataset and [pixelNeRF](https://alexyu.net/pixelnerf/) details.
We thank Prof. [Alex Berg](http://acberg.com/) of UNC for the generous offer of computational resources and [Misha Shvets](https://scholar.google.com/citations?user=CY9PcHEAAAAJ&hl=en) of UNC
for a useful tutorial on it. This work was partially supported by National Science Foundation Award 1840131.
## Contact

For questions, please contact [youngjoong@cs.unc.edu](youngjoong@cs.unc.edu).
