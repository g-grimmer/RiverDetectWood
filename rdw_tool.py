#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import subprocess
import rasterio
from rasterio.enums import Compression
from scipy.ndimage import label, find_objects
from collections import Counter
import joblib
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
from shapely.geometry import Polygon
import math
import os
import platform
from os.path import join
from osgeo import gdal, ogr, osr

import warnings
warnings.filterwarnings("ignore")

def calculate_ndvi(input_tiff):
    """
    Calculate the NDVI (Normalized Difference Vegetation Index) from a given input TIFF.

    Args:
        input_tiff (str): Path to the input TIFF file.

    Returns:
        np.array: NDVI array.
    """
    with gdal.config_option("CHECK_DISK_FREE_SPACE", "NO"):
        dataset = gdal.Open(input_tiff)
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        ndvi_array = np.zeros((ysize, xsize), dtype=np.float32)
        block_size = 1024

        for y in range(0, ysize, block_size):
            y_block_size = min(block_size, ysize - y)
            for x in range(0, xsize, block_size):
                x_block_size = min(block_size, xsize - x)
                band_nir = dataset.GetRasterBand(1).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                band_red = dataset.GetRasterBand(2).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                ndvi = (band_nir - band_red) / (band_nir + band_red + 1e-10)
                ndvi_array[y:y + y_block_size, x:x + x_block_size] = ndvi
        dataset = None
        return ndvi_array

def calculate_brightness(input_tiff):
    """
    Calculate the brightness index from the input TIFF.

    Args:
        input_tiff (str): Path to the input TIFF file.

    Returns:
        np.array: Brightness array.
    """
    with gdal.config_option("CHECK_DISK_FREE_SPACE", "NO"):
        dataset = gdal.Open(input_tiff)
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        brightness_array = np.zeros((ysize, xsize), dtype=np.float32)
        block_size = 1024

        for y in range(0, ysize, block_size):
            y_block_size = min(block_size, ysize - y)
            for x in range(0, xsize, block_size):
                x_block_size = min(block_size, xsize - x)
                band_nir = dataset.GetRasterBand(1).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                band_red = dataset.GetRasterBand(2).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                brightness = np.sqrt(band_red**2 + band_nir**2)
                brightness_array[y:y + y_block_size, x:x + x_block_size] = brightness
        dataset = None
        return brightness_array

def calculate_texture(input_tiff, output_tiff):
    """
    Calculate texture indices (Haralick) using the Orfeo Toolbox (OTB).

    Args:
        input_tiff (str): Path to the input TIFF.
        output_tiff (str): Path to save the output texture TIFF.

    Raises:
        EnvironmentError: If an error occurs when detecting the operating system.
        OSError: If OTB is not supported on the current OS.

    Returns:
        np.array: Array of texture bands.
    """
    otb_path = os.getenv('OTB_BIN_PATH')
    
    # Detect the operating system
    os_type = platform.system()

    if os_type == "Windows":
        command = [
            os.path.join('otbcli_HaralickTextureExtraction.bat'),
            "-in", input_tiff,
            "-channel", "2",
            "-parameters.xrad", "3",
            "-parameters.yrad", "3",
            "-texture", "simple",
            "-out", output_tiff
        ]
    elif os_type == "Linux":
        command = [
            'otbcli_HaralickTextureExtraction',
            "-in", input_tiff,
            "-channel", "2",
            "-parameters.xrad", "3",
            "-parameters.yrad", "3",
            "-texture", "simple",
            "-out", output_tiff
        ]
    else:
        raise OSError(f"Unsupported operating system: {os_type}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("Command executed successfully.")
        print("Standard output:", result.stdout)
        dataset = gdal.Open(output_tiff)
        if dataset is None:
            print("Error: the texture file was not created correctly.")
            return None
        texture_band = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        dataset = None
        return texture_band
    except subprocess.CalledProcessError as e:
        print("Error while executing the command.")
        print("Return code:", e.returncode)
        print("Standard error:", e.stderr)
        return None

def merge_indices_with_input(input_tiff, ndvi, brightness, texture_tiff, output_tiff):
    """
    Merge the input image with calculated NDVI, brightness, and texture indices.

    Args:
        input_tiff (str): Path to the input TIFF.
        ndvi (np.array): NDVI array.
        brightness (np.array): Brightness array.
        texture_tiff (str): Path to the texture TIFF.
        output_tiff (str): Path to save the merged output TIFF.
    """
    input_dataset = gdal.Open(input_tiff)
    xsize = input_dataset.RasterXSize
    ysize = input_dataset.RasterYSize
    num_bands = input_dataset.RasterCount
    texture_dataset = gdal.Open(texture_tiff)
    if texture_dataset is None:
        print("Error: the texture file was not opened correctly.")
        return
    homogeneity_band = texture_dataset.GetRasterBand(4).ReadAsArray().astype(np.float32)
    entropy_band = texture_dataset.GetRasterBand(2).ReadAsArray().astype(np.float32)
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_tiff, xsize, ysize, num_bands + 4, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
    out_dataset.SetProjection(input_dataset.GetProjection())

    for i in range(1, num_bands + 1):
        band_data = input_dataset.GetRasterBand(i).ReadAsArray().astype(np.float32)
        out_band = out_dataset.GetRasterBand(i)
        out_band.WriteArray(band_data)

    out_band = out_dataset.GetRasterBand(num_bands + 1)
    out_band.WriteArray(ndvi)
    out_band = out_dataset.GetRasterBand(num_bands + 2)
    out_band.WriteArray(homogeneity_band)
    out_band = out_dataset.GetRasterBand(num_bands + 3)
    out_band.WriteArray(brightness)
    out_band = out_dataset.GetRasterBand(num_bands + 4)
    out_band.WriteArray(entropy_band)

    for i in range(1, num_bands + 4):
        out_dataset.GetRasterBand(i).SetNoDataValue(255)

    out_dataset.FlushCache()
    input_dataset = None
    texture_dataset = None
    out_dataset = None

from tqdm import tqdm  # Import tqdm for the progress bar

def classify_correct_vectorize(raster_path, model_path, output_corrected_path, output_shapefile, value_to_keep=1, min_size=400, tile_size=512):
    """
    Classify the raster using a Random Forest model, correct and vectorize the output, with optimizations to skip no-data areas and process in tiles.

    Args:
        raster_path (str): Path to the merged raster.
        model_path (str): Path to the Random Forest model.
        output_corrected_path (str): Path to save the corrected output.
        output_shapefile (str): Path to save the vectorized output.
        value_to_keep (int, optional): Value to retain in the output. Defaults to 1.
        min_size (int, optional): Minimum polygon size to retain. Defaults to 400.
        tile_size (int, optional): Size of the tiles (in pixels) to process independently. Defaults to 512.
    """
    
    # Open the raster and model
    raster_ds = gdal.Open(raster_path)
    model = joblib.load(model_path)
    
    # Get raster dimensions and initialize output array
    num_rows, num_cols = raster_ds.RasterYSize, raster_ds.RasterXSize
    predicted_image = np.zeros((num_rows, num_cols), dtype=np.uint8)
    
    # Calculate total number of tiles for the progress bar
    total_tiles = ((num_rows + tile_size - 1) // tile_size) * ((num_cols + tile_size - 1) // tile_size)
    
    # Process the raster in tiles with a progress bar
    with tqdm(total=total_tiles, desc="Processing tiles", unit="tile") as pbar:
        for y in range(0, num_rows, tile_size):
            for x in range(0, num_cols, tile_size):
                y_block_size = min(tile_size, num_rows - y)
                x_block_size = min(tile_size, num_cols - x)
                
                # Read and stack bands for the tile
                bands = [raster_ds.GetRasterBand(i+1).ReadAsArray(x, y, x_block_size, y_block_size) for i in range(7)]
                stacked_bands = np.stack(bands, axis=2)
                
                # Create a no-data mask for the tile
                no_data_mask = (stacked_bands[:, :, 0] == 0) & (stacked_bands[:, :, 1] == 0) & (stacked_bands[:, :, 2] == 0)
                
                # Flatten non-masked pixels and classify
                masked_pixels = stacked_bands[~no_data_mask]
                if masked_pixels.size == 0:
                    pbar.update(1)
                    continue  # Skip if the tile is entirely no-data
                
                pixels = masked_pixels.reshape(-1, stacked_bands.shape[2])
                predictions = model.predict(pixels)
                
                # Reintegrate predictions into the predicted image array
                tile_prediction = np.zeros((y_block_size, x_block_size), dtype=np.uint8)
                tile_prediction[~no_data_mask] = predictions
                predicted_image[y:y + y_block_size, x:x + x_block_size] = tile_prediction

                # Update the progress bar after processing each tile
                pbar.update(1)

    # Save the corrected image in a new raster
    driver = gdal.GetDriverByName("GTiff")
    corrected_ds = driver.Create(output_corrected_path, num_cols, num_rows, 1, gdal.GDT_Byte)
    corrected_ds.SetGeoTransform(raster_ds.GetGeoTransform())
    corrected_ds.SetProjection(raster_ds.GetProjection())
    
    out_band = corrected_ds.GetRasterBand(1)
    out_band.WriteArray(predicted_image)
    out_band.SetNoDataValue(0)
    
    corrected_ds.FlushCache()
    corrected_ds = None 

    # Open the corrected file for vectorization
    corrected_ds = gdal.Open(output_corrected_path)
    band = corrected_ds.GetRasterBand(1)
    
    # Create the output shapefile
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile_ds = shp_driver.CreateDataSource(output_shapefile)
    
    # Set shapefile projection
    srs = osr.SpatialReference()
    srs.ImportFromWkt(corrected_ds.GetProjection())
    
    # Create polygon layer
    layer = shapefile_ds.CreateLayer(output_shapefile, srs=srs, geom_type=ogr.wkbPolygon)
    field_defn = ogr.FieldDefn("Class", ogr.OFTInteger)
    layer.CreateField(field_defn)
    
    # Vectorize with GDAL Polygonize
    gdal.Polygonize(band, None, layer, 0, [], callback=None)
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        class_value = feature.GetFieldAsInteger("Class")
        
        # Check if the class is 1 and area is larger than min_size
        if class_value != value_to_keep:
            layer.DeleteFeature(feature.GetFID())
    
    # Close files
    shapefile_ds = None
    corrected_ds = None
    raster_ds = None

    print(f"Polygons with value {value_to_keep} have been saved to {output_shapefile}")


def calculer_volume(shapefile_path):
    """
    Calculate volume metrics for polygons in a shapefile.

    Args:
        shapefile_path (str): Path to the shapefile.

    Returns:
        str: Path to the updated shapefile with calculated metrics.
    """
    gdf = gpd.read_file(shapefile_path)

    def calculer_aire(poly):
        return poly.area

    def calculer_emprise_minimum(poly):
        convex_hull = poly.convex_hull
        min_rect = convex_hull.minimum_rotated_rectangle
        bounds = min_rect.bounds
        longueur = bounds[2] - bounds[0]
        largeur = bounds[3] - bounds[1]
        aire_emprise = min_rect.area
        return longueur, largeur, aire_emprise

    def calculer_volume_polygone(poly, aire_polygone, longueur_emprise, largeur_emprise, aire_emprise):
        c = aire_polygone / aire_emprise
        volume = c * longueur_emprise * (largeur_emprise ** 2) * (math.pi / 4)
        return volume

    gdf['Polygon_Area'] = gdf.geometry.apply(calculer_aire)
    gdf['Bounding_Length'], gdf['Bounding_Width'], gdf['Bounding_Area'] = zip(*gdf.geometry.apply(calculer_emprise_minimum))
    gdf['Volume'] = gdf.apply(lambda row: calculer_volume_polygone(row.geometry, row['Polygon_Area'], row['Bounding_Length'], row['Bounding_Width'], row['Bounding_Area']), axis=1)
    gdf['Corrected_Area'] = 0.49 * gdf['Polygon_Area'] ** 1.12
    gdf['Corrected_Length'] = 0.88 * gdf['Bounding_Length'] ** 1.02
    gdf['Corrected_Width'] = 0.48 * gdf['Bounding_Width'] ** 1.24
    gdf['Corrected_Volume'] = 0.24 * gdf['Volume'] ** 1.17

    apply_filter = input("Would you like to apply a filter to remove polygons whose length or width exceeds a certain threshold? (yes/no):")
    if apply_filter.lower() == 'yes':
        threshold = float(input("Please enter the threshold size:"))
        gdf = gdf[(gdf['Corrected_Length'] <= threshold) & (gdf['Corrected_Width'] <= threshold)]
    gdf.drop(columns=['Bounding_Area', 'Polygon_Area', 'Bounding_Length', 'Bounding_Width', 'Volume'], inplace=True)
    gdf.rename(columns={
        'Corrected_Area': 'Area',
        'Corrected_Length': 'Length',
        'Corrected_Width': 'Width',
        'Corrected_Volume': 'Volume'
    }, inplace=True)
    output_shapefile = shapefile_path.replace('.shp', '_metric.shp')
    gdf.to_file(output_shapefile)
    return output_shapefile

def main():
    """
    Main function to process satellite images and calculate indices, classify, correct,
    vectorize and compute volume metrics.
    """
    parser = argparse.ArgumentParser(description="Process satellite images to extract and calculate various indices, classify, correct, vectorize and compute volume metrics.")
    parser.add_argument("input_tiff", help="Input TIFF file path.")
    parser.add_argument("model_path", help="Model path for classification.")
    parser.add_argument("keep_all_outputs", help="1 to keep all outputs, 0 to delete them")

    args = parser.parse_args()
    output_folder = join(os.getcwd(), 'outputs')
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    keep_all_outputs = True if args.keep_all_outputs == '1' else False
    output_shapefile = join(output_folder, 'deadwood_binary.shp')
    shapefile_path = join(output_folder, 'deadwood_metrics.shp')
    
    output_tiff_text = join(output_folder, 'Haralick_textural_indices.tif')
    output_tiff_merge = join(output_folder, 'full_stack.tif')
    output_corrected_path = join(output_folder, 'full_stack_corrected.tif')
    
    print("[INFO] Calculating NDVI ...")
    ndvi = calculate_ndvi(args.input_tiff)
    print("[INFO] NDVI successfully calculated.")
    
    print("[INFO] Calculating BI ...")
    brightness = calculate_brightness(args.input_tiff)
    print("[INFO] BI successfully calculated.")
    
    print("[INFO] Calculating Haralick ...")
    calculate_texture(args.input_tiff, output_tiff_text)
    print("[INFO] Haralick successfully calculated.")
    
    print("[INFO] Classification ... This can take several minutes depending on your image size.")
    merge_indices_with_input(args.input_tiff, ndvi, brightness, output_tiff_text, output_tiff_merge)
    classify_correct_vectorize(output_tiff_merge, args.model_path, output_corrected_path, output_shapefile, value_to_keep=1, min_size=400)
    print("[INFO] Classification finished.")
    
    print("[INFO] Calculating Deadwood metrics ...")
    calculer_volume(output_shapefile)
    print("[INFO] Deadwood metrics successfully calculated.")
    
    if not keep_all_outputs:
        os.remove(output_tiff_text)
        os.remove(output_tiff_merge)
        os.remove(output_corrected_path)
    
    print("[OUTPUTS] Deadwood binary: " + str(output_shapefile))
    print("[OUTPUTS] Deadwood metrics: " + str(shapefile_path))

if __name__ == "__main__":
    main()
