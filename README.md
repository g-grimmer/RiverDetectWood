# RiverDetectWood

This Python script allows to detect river dead wood using IGN BD Ortho data. Firstly, it calculates indexes to be added to orthophotographs. Secondly it applies the random forest model on the orthophotographs. Finally, it output a shapefile layer with dead wood polygons and their metrics. 
This tool only works on Windows or Linux

## Clone the repository

Firstly you need to clone the repository :

```bash
git clone https://github.com/g-grimmer/RiverDetectWood.git
cd RiverDetectWood
```

## Installation

### Prerequisites

- Python 3.x
- `GDAL`
- `Rasterio`
- `Scipy`
- `Joblib`
- `Geopandas`
- Orfeo Toolbox (OTB) version 9.x

### Install Python Packages

We recommand to use Anaconda.

```bash
conda create -n myenv python=3.9
conda activate myenv
conda install numpy gdal rasterio shapely geopandas scipy scikit-learn joblib
```

### Install Orfeo Toolbox (OTB)

#### Install OTB from website

Firstly you have to install OTB from https://www.orfeo-toolbox.org/download/

#### Set environment variable

Here you must specify the path to the file otbcli_HaralickTextureExtraction.bat as follows. 
This file is contained in the "bin" directory :

On Windows 

```bash
set OTB_BIN_PATH=C:\path\to\otb\bin\otbcli_HaralickTextureExtraction.bat"
```

On Linux

On linux, you can follow the steps describe on OTB official website. Basically, you just need to source the environment profile before launching our script on the same terminal.

```bash
source /Path/To/OTB_install/otbenv.profile
```
## Usage

### Downlead model file

In this repository the model file is compressed.
You will need to decompress it and specify its path in the tool

### RiverDetectTool application

```bash
python rdw_tool.py input_tiff_path output_tiff_text_path output_tiff_merge_path model_path output_corrected_path output_shapefile_path shapefile_path
```
In this function :
- `input_tiff_path` is the orthophotograph path
- `output_tiff_text_path` is the path of various textures calculated on the orthophotograph
- `output_tiff_merge_path` is the path of orthophotograph that contains all indexes
- `model_path`is the random forest model file path
- `output_corrected_path` is the classification result path
- `output_shapefile_path` is the path of shapefile layer that contains the dead wood polygons without metrics
- `shapefile_path`is the path of final shapefile layer that contains the dead wood polygons with their metrics


