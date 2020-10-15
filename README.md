 # Video Frame Interpolation without Temporal Priors (NeurIPS2020)
[Paper]
<!--  
Conference   
-->   
</div>
  

## How to run   
### Prerequisites
- NVIDIA GPU + CUDA CuDNN 
- Pytorch

```bash
# clone project   
git clone https://github.com/yjzhang96/UTI-VFI 
cd UTI-VFI
 ``` 
### Dataset
download GoPro datasets, and generate blurry videos for different exposure settings.
   

### Test
 ```bash
sh run_test.sh
```


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
