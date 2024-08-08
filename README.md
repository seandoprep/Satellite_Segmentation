> Code is still on working. Please wait for final version.

# 🛰️Satellite_Segmentation🛰️ : Segmentation Framework Using PyTorch for Satellite Imagery.

## About the Project
- This code is originally built for segmenting offshore aquaculture facilities in the Southern Sea of South Korea, but you can just apply this pipeline into any kind of satellite segmentation tasks. 
- You can get your target segmentation result with non-processed Satellite Imagery and target mask.
- Code will be updated constantly. If you have any advice or questions, please feel free to contact me via email(sean3819@yonsei.ac.kr).

## Requirements
</br>

### Conda virtual environment setup (recommend python 3.9.18.ver )

```
conda create -n [environment name] --file [this file]
conda activate [environment name]
```
</br>

### Git clone repo

```
git clone https://github.com/seandoprep/Satellite_Segmentation.git
```

Clone this repo into your own computer. You can just create data directory by yourself. 

</br>

And finally the directory hierarchy is configured as,

```
Satellite_Segmentation
├─data
│  └─Train
│      └─ENVI
│          ├─Image
│          ├─Mask
│          └─original_nc
├─models
│  ├─modules
│  └─models
├─outputs
│  ├─inference_output
│  ├─test_output
│  └─train_output
├─utils
│  ├─metrics.py
│  ├─save_data.py
│  ├─util.py
│  └─visualize.py
├─weights
│  └─train_result
├─band_analysis.py
├─dataset.py
├─loss.py
├─scheduler.py
├─train.py
├─test.py
├─inference.py
└─requirements.txt

```

---

## Dataset
</br>

### Data Format
- Currently, this code can only handle ENVI type Satellite images. 
- Code for handling other types of data(netcdf, tiff, etc..) will also be added
- Data should be large original satellite image without cutting. Built-in algorithm will divide Large original images into 256 width, height images.
- original nc data needs for extracting lat/lon information.
- Using all of the huge satellite data can lead to a data distribution imbalance problem. 
- Therefore, after sampling the deep learning data based on the target mask, the data of the non-target area may be randomly added.

```
data
├── Train
│      ├── ENVI
│      │      ├── Image
│      │      │      ├── NDWI.hdr
│      │      │      ├── NDWI.img
│      │      │      ├── NIR.hdr
│      │      │      ├── NIR.img
│      │      │      └── .....
│      │      ├── Mask
│      │      │      ├── Mask.hdr
│      │      │      └── Mask.img
│      │      ├── original_nc
└──    └──    └──    └── original.nc

```
</br>

### Sample data(will be added)



## Train/Test/Inference

### Supported model

- In order to handle multi-channel data, I add some feature extractor block in front of some models. 
- U-Net, U2-Net, DeepLabv3+, ResUNet++, Attention U-Net, Mdoaunet, RaftNet are now available

### Train
</br>
python train.py -D [data directory] -M [model name] -E [num epochs] -L [learning rate] -B [batch size] -S [early stop]
</br>

### Test
</br>
python test.py -D [data directory] -M [model name] -P [model path]
</br>

### Inference
</br>
python train.py -D [data directory] -M [model name] -P [model path] -B [batch size]
</br>

### Performance(will be added) : 
</br>


## Qualitative results(will be added)

</br>


## Reference(will be added)
Thanks to,
https://github.com/usuyama/pytorch-unet
https://github.com/xuebinqin/U-2-Net
https://github.com/rishikksh20/ResUnet
https://github.com/LeeJunHyun/Image_Segmentation
https://github.com/mukund-ks/DeepLabV3Plus-PyTorch
https://github.com/Jichao-Wang/MDOAU-net/blob/main/MDOAU_net.py
https://www.mdpi.com/2072-4292/14/18/4587