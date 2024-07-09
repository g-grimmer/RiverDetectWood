# RiverDetectWood

This Python script allows to detect river dead wood using IGN BD Ortho data. Firstly, it calculates indexes to be added to orthophotographs. Secondly it applies the random forest model on the orthophotographs. Finally, it output a shapefile layer with dead wood polygons and their metrics. 

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
python rdw_script.py input_tiff_path output_tiff_text_path output_tiff_merge_path model_path output_corrected_path output_shapefile_path shapefile_path
```



