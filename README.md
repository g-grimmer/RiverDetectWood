# RiverDetectWood

This Python script allows to detect river dead wood using IGN BD Ortho data. Firstly, it calculates indexes to be added to orthophotographs. Secondly it applies the random forest model on the orthophotographs. Finally, it output a shapefile layer with dead wood polygons and their metrics. 

## Clone the repository

Firstly you need to clone the repository :

```bash
git clone https://github.com/g-grimmer/RiverDetectWood.git
cd RiverDetectWood
```

## Git Large File Storage

### Install Git LFS

You need to install Git LFS to handle the random forest model file, which is essential for the tool to function properly.

If you have not install Git LFS yet, you can follow the instructions on https://docs.github.com/fr/repositories/working-with-files/managing-large-files/installing-git-large-file-storage

### Download model file from Git LFS

Retrive file : 

```bash
git lfs fetch
```

File checking :

```bash
git lfs ls-files
```

## Installation

### Prerequisites

- Python 3.x
- `GDAL`
- `Rasterio`
- `Scipy`
- `Joblib`
- `Geopandas`
- Orfeo Toolbox (OTB)

### Install Python Packages

```bash
pip install gdal rasterio scipy joblib geopandas
```

## Install Orfeo Toolbox (OTB)

### Install OTB from website

Firstly you have to install OTB from https://www.orfeo-toolbox.org/download/

### Set environment variable

On Windows 

```bash
set OTB_BIN_PATH=C:\path\to\otb\bin
```

On Linux

```bash
export OTB_BIN_PATH=/path/to/otb/bin
```

On MacOS

```bash
```

## Usage

```bash
python rdw_tool.py input_tiff_path output_tiff_text_path output_tiff_merge_path model_path output_corrected_path output_shapefile_path shapefile_path
```



