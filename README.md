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

```sh
pip install gdal rasterio scipy joblib geopandas
```

## Install Orfeo Toolbox (OTB)

### Install OTB from website

Firstly you have to install 

