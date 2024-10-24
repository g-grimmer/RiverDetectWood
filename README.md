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

- Python 3.9
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
conda install -c conda-forge numpy gdal rasterio shapely geopandas scipy scikit-learn joblib
```

### Install Orfeo Toolbox (OTB)

#### Install OTB from website

Firstly you have to install OTB from https://www.orfeo-toolbox.org/download/

#### Execute otbenv.bat

Here you must execute the file otbenv.bat (or otbenv.profile on linux):

On Windows 

```bash
C:\path\to\otb\bin\otbenv.bat"
```

On Linux

On linux, you can follow the steps describe on OTB official website. Basically, you just need to source the environment profile before launching our script on the same terminal.

```bash
source /Path/To/OTB_install/otbenv.profile
```
## Usage

### Download model and test files

You can download the model file (.rf) and the test file (.tif) from this link: ...

### RiverDetectTool application

```bash
python rdw_tool.py input_tiff_path output_tiff_text_path keep_all_outputs
```
In this function :
- `input_tiff_path` is the orthophotograph path
- `model_path`is the random forest model file path
- `keep_all_outputs`, 1 to keep all outputs, 0 to delete them

## Citation

If you use this repository in your research, please cite the corresponding paper:

```
@article{grimmer2024deadwood,
  title={...},
  author={Grimmer, Gauthier and Wenger, Romain and Chardon, Valentin},
  journal={SoftwareX},
  year={2024},
  publisher={Elsevier}
}
```
