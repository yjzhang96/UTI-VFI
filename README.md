# Video Frame Interpolation without Temporal Priors (NeurIPS2020)

[Paper]
<!--  
Conference   
-->   
</div>


## How to run

### Prerequisites

- NVIDIA GPU + CUDA 9.0 + CuDNN 7.6.5
- Pytorch 1.1.0


First clone the project   
```bash
git clone https://github.com/yjzhang96/UTI-VFI 
cd UTI-VFI
 ``` 
Download pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1kW8v66c1_FAyi_qAojJ6SjBop8ufduSY?usp=sharing).
Put model weights "SEframe_net.pth" and "refine_net.pth" into directory "./UTI_model_weights"; put "model.ckpt" and "network-default.pytorch" into directory "./utils"

### Dataset
download GoPro datasets with all the figh-frame-rate video frames from [GOPRO_Large_all](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view), and generate blurry videos for different exposure settings.


### Test
 ```bash
sh run_test.sh
```
</div>


## Citation   
```
@article{Zhang2019video,
  title={Video Frame Interpolation without Temporal Priors},
  author={Zhang, Youjian and Wang, Chaoyue and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```
## Acknowledgment
Code of interpolation module borrows heavily from [QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)
