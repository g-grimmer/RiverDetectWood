# RiverDetectWood

This Python script allows to detect river dead wood using IGN BD Ortho data. Firstly, it calculates indexes to be added to orthophotographs. Secondly it applies the random forest model on the orthophotographs. Finally, it output a shapefile layer with dead wood polygons and their metrics. 

## Tool application

```bash
python satellite_processing_tool.py input_tiff_path output_tiff_text_path output_tiff_merge_path model_path output_corrected_path output_shapefile_path shapefile_path
```
