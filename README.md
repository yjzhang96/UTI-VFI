# Video Frame Interpolation without Temporal Priors (NeurIPS2020)

[[Paper](https://github.com/yjzhang96/UTI-VFI/raw/master/paper/nips_camera_ready.pdf)] [[video](https://drive.google.com/file/d/1eRUzA2m3EvvrHw0MhO4AMxEQEgdt8LGc/view?usp=sharing)]
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
mkdir pretrain_models
 ``` 
Download pretrained model weights from [Google Drive](https://drive.google.com/drive/folders/1kW8v66c1_FAyi_qAojJ6SjBop8ufduSY?usp=sharing).
Put model weights "SEframe_net.pth" and "refine_net.pth" into directory "./pretrain_models"; put "model.ckpt" and "network-default.pytorch" into directory "./utils"

### Dataset
download GoPro datasets with all the figh-frame-rate video frames from [GOPRO_Large_all](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view), and generate blurry videos for different exposure settings. You can generate the test datasets via run:
```bash
python utils/generate_blur.py
```


### Test
After prepared test datasets, you can run test usding the following command:
 ```bash
sh run_test.sh
```
Note that to test the model on GOPRO datasets (datasets with groud-truth to compare), you need to set the argument "--test_type" to ''validation''. If you want to test the model on real-world video (without ground-truth), you need to use "real_world" instead.
</div>


## Citation   
```
@inproceedings{Zhang2019video,
  title={Video Frame Interpolation without Temporal Priors},
  author={Zhang, Youjian and Wang, Chaoyue and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
```
## Acknowledgment
Code of interpolation module borrows heavily from [QVI](https://sites.google.com/view/xiangyuxu/qvi_nips19)
