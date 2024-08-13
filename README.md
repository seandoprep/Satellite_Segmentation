> Code is still on working. Please wait for final version.

# 🛰️Satellite_Segmentation🛰️ : Segmentation Framework Using PyTorch for Satellite Imagery.

## About the Project
- This code is originally built for segmenting offshore aquaculture facilities in the Southern Sea of South Korea, but you can just apply this pipeline into any kind of satellite segmentation tasks. 
- You can get your target segmentation result with non-processed Satellite Imagery and target mask.
- Code will be updated constantly. If you have any advice or questions, please feel free to contact me via email(sean3819@yonsei.ac.kr).
- Flowchart :
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/flowchart.png?raw=true">

## Requirements
### Conda virtual environment setup (recommend python 3.9.18.ver)
```
conda create -n [environment name] --file [this file]
conda activate [environment name]
```

### Git clone repo

```
git clone https://github.com/seandoprep/Satellite_Segmentation.git
```

Clone this repo into your own computer. You can just create data directory by yourself. 

And finally the directory hierarchy is configured as,

</br>

```
Satellite_Segmentation
├─data
│  └─Train
│   ├─Image
│   ├─Mask
│   └─original_nc
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
### Data Format
- Currently, this code can handle ENVI, Tiff, Tif, NetCDF type Satellite images. 
- Data should be large original satellite image without cutting. Built-in algorithm will divide original images into 256 width, height images.
- original nc data needs for extracting lat/lon information.
- For visualizing train process, you should modify band_names variable in visualize.py as your need
- Using all of the huge satellite data can lead to a data distribution imbalance problem. 
- Therefore, after sampling the deep learning data based on the target mask, the data of the non-target area(Where target object doesn't exist) may be randomly added.

```
data
├── Train
│      ├── Image
│      │      ├── NDWI.hdr
│      │      ├── NDWI.img
│      │      ├── NIR.hdr
│      │      ├── NIR.img
│      │      └── .....
│      ├── Mask
│      │      ├── Mask.hdr
│      │      └── Mask.img
│      └── original_nc
└──           └── original.nc

```

### Sample data
Data Download : [Google Drive](https://drive.google.com/drive/folders/1skobdYIpn0LS9YqQvlGoWnWkUhu5otas?usp=sharing)
- Task : Aquaculture Facility Segmentation
- Satellite Info : Sentinel-2A/B 
- Spatial Resolution : 10m
- Bands : NDWI, NIR, RED
- True Mask was created based on a Google Earth and aquculture facility information map provided by the Ministry of Oceans and Fisheries.

---
## Band Analysis
- Visualizing Normalization result. Other analysis technique like band histogram will be updated soon.
```
python band_analysis.py -D [data directory] -T [visualize type : norm]
```

### Normalization Result
- Linear Norm & Dynamic World Norm are now available
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/norm_result.png?raw=true">

---
## Train/Test/Inference

### Supported model
- In order to handle multi-channel data, I add some feature extractor block in front of some models. 
- U-Net, U2-Net, DeepLabv3+, ResUNet++, Attention U-Net, Mdoaunet, RaftNet are now available

### Train
```
python train.py -D [data directory] -M [model name] -E [num epochs] -L [learning rate] -B [batch size] -S [early stop]
```
### Test
```
python test.py -D [data directory] -M [model name] -P [model path]
```
### Inference
```
python train.py -D [data directory] -M [model name] -P [model path] -B [batch size]
```
---
## Qualitative results
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/result1.png?raw=true">
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/result2.png?raw=true">

### Segmentation Map(Multipolygon Shapefile format)
- Inference_output.jpg : Inference result for original data(total segmentation map)
- Compare_output.jpg : Compare inference result with true mask(red : true mask, blue : inference result)
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/inference_result.png?raw=true">


### Hexagonal Distribution
- Geocoordinated hexagonal distribution can be obtained using the plt.hexbin function.
- Use visualization tools such as QGIS
<img src="https://github.com/seandoprep/Satellite_Segmentation/blob/master/images/hexbin_distribution.png?raw=true">

---
## Code Reference
Thanks to,
</br>
https://github.com/usuyama/pytorch-unet
https://github.com/xuebinqin/U-2-Net
https://github.com/rishikksh20/ResUnet
https://github.com/LeeJunHyun/Image_Segmentation
https://github.com/mukund-ks/DeepLabV3Plus-PyTorch
https://github.com/Jichao-Wang/MDOAU-net/blob/main/MDOAU_net.py
https://www.mdpi.com/2072-4292/14/18/4587