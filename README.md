> Code is still on working. Please wait for final version.

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

You can just clone this repo into your own computer

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
- Currently, this code can only handle ENVI type Satellite images. Code for handling other types of data(netcdf, tiff, etc..) will also be added
- Recommended input image resolution is (256, 256).
- Data should be large original satellite image without cutting. Built-in algorithm will divide Large original images into 256 width, height images.
- nc data needs for extracting lat/lon information.

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

## Train(To be written
</br>

### Supported model

- In order to handle multi-channel data, I add some feature extractor block in front of some models. 

</br>

### Train code

</br>

## Evaluation

### Performance : 
</br>


## Qualitative results

</br>
