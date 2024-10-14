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

def calculate_ndvi(input_tiff):
    """_summary_

    Args:
        input_tiff (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        input_tiff (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        input_tiff (_type_): _description_
        output_tiff (_type_): _description_

    Raises:
        EnvironmentError: _description_
        OSError: _description_

    Returns:
        _type_: _description_
    """
    otb_path = os.getenv('OTB_BIN_PATH')
    

    # Detect the operating system
    os_type = platform.system()

    if os_type == "Windows":
        if not otb_path:
            raise EnvironmentError("OTB_BIN_PATH environment variable is not set.")
        command = [
            os.path.join(otb_path, 'otbcli_HaralickTextureExtraction.bat'),
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
        print("Commande exécutée avec succès")
        print("Sortie standard:", result.stdout)
        dataset = gdal.Open(output_tiff)
        if dataset is None:
            print("Erreur : le fichier de texture n'a pas été créé correctement.")
            return None
        texture_band = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        dataset = None
        return texture_band
    except subprocess.CalledProcessError as e:
        print("Erreur lors de l'exécution de la commande")
        print("Code de retour:", e.returncode)
        print("Erreur standard:", e.stderr)
        return None


def merge_indices_with_input(input_tiff, ndvi, brightness, texture_tiff, output_tiff):
    """_summary_

    Args:
        input_tiff (_type_): _description_
        ndvi (_type_): _description_
        brightness (_type_): _description_
        texture_tiff (_type_): _description_
        output_tiff (_type_): _description_
    """
    input_dataset = gdal.Open(input_tiff)
    xsize = input_dataset.RasterXSize
    ysize = input_dataset.RasterYSize
    num_bands = input_dataset.RasterCount
    texture_dataset = gdal.Open(texture_tiff)
    if texture_dataset is None:
        print("Erreur : le fichier de texture n'a pas été ouvert correctement.")
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


def classify_correct_vectorize(raster_path, model_path, output_corrected_path, output_shapefile, value_to_keep=1, min_size=400):
    """_summary_

    Args:
        raster_path (_type_): _description_
        model_path (_type_): _description_
        output_corrected_path (_type_): _description_
        output_shapefile (_type_): _description_
        value_to_keep (int, optional): _description_. Defaults to 1.
        min_size (int, optional): _description_. Defaults to 400.
    """
    
    # Ouverture du raster et du modèle
    raster_ds = gdal.Open(raster_path)
    band1 = raster_ds.GetRasterBand(1)
    band2 = raster_ds.GetRasterBand(2)
    band3 = raster_ds.GetRasterBand(3)
    band4 = raster_ds.GetRasterBand(4)
    band5 = raster_ds.GetRasterBand(5)
    band6 = raster_ds.GetRasterBand(6)
    band7 = raster_ds.GetRasterBand(7)
    
    # Empilement des bandes et classification avec un modèle
    stacked_bands = np.stack((band1.ReadAsArray(), band2.ReadAsArray(), band3.ReadAsArray(),
                              band4.ReadAsArray(), band5.ReadAsArray(), band6.ReadAsArray(), 
                              band7.ReadAsArray()), axis=2)
    num_rows, num_cols, num_bands = stacked_bands.shape
    pixels = stacked_bands.reshape(num_rows * num_cols, num_bands)
    
    # Chargement du modèle Random Forest et prédiction
    rf_optimized = joblib.load(model_path)
    predictions = rf_optimized.predict(pixels)
    predicted_image = predictions.reshape(num_rows, num_cols)

    # Sauvegarde de l'image corrigée dans un nouveau raster
    driver = gdal.GetDriverByName("GTiff")
    corrected_ds = driver.Create(output_corrected_path, num_cols, num_rows, 1, gdal.GDT_Byte)
    corrected_ds.SetGeoTransform(raster_ds.GetGeoTransform())
    corrected_ds.SetProjection(raster_ds.GetProjection())
    
    out_band = corrected_ds.GetRasterBand(1)
    out_band.WriteArray(predicted_image)
    out_band.SetNoDataValue(0)
    
    corrected_ds.FlushCache()
    corrected_ds = None 

    # Ouverture du fichier corrigé pour vectorisation
    corrected_ds = gdal.Open(output_corrected_path)
    band = corrected_ds.GetRasterBand(1)
    
    # Création du shapefile de sortie
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile_ds = shp_driver.CreateDataSource(output_shapefile)
    
    # Définir la projection du shapefile
    srs = osr.SpatialReference()
    srs.ImportFromWkt(corrected_ds.GetProjection())
    
    # Création de la couche de polygones
    layer = shapefile_ds.CreateLayer(output_shapefile, srs=srs, geom_type=ogr.wkbPolygon)
    field_defn = ogr.FieldDefn("Class", ogr.OFTInteger)
    layer.CreateField(field_defn)
    
    # Vectorisation avec GDAL Polygonize
    gdal.Polygonize(band, None, layer, 0, [], callback=None)
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        class_value = feature.GetFieldAsInteger("Class")
        
        # Vérification si la classe est 1 et que la surface est supérieure au min_size
        if class_value != 1:
            layer.DeleteFeature(feature.GetFID())
    
    # Fermeture des fichiers
    shapefile_ds = None
    corrected_ds = None
    raster_ds = None

    print(f"Les polygones avec la valeur {value_to_keep} ont été sauvegardés dans {output_shapefile}")



def calculer_volume(shapefile_path):
    """_summary_

    Args:
        shapefile_path (_type_): _description_

    Returns:
        _type_: _description_
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

    gdf['Aire_Polygone'] = gdf.geometry.apply(calculer_aire)
    gdf['Longueur_Emprise'], gdf['Largeur_Emprise'], gdf['Aire_Emprise'] = zip(*gdf.geometry.apply(calculer_emprise_minimum))
    gdf['Volume'] = gdf.apply(lambda row: calculer_volume_polygone(row.geometry, row['Aire_Polygone'], row['Longueur_Emprise'], row['Largeur_Emprise'], row['Aire_Emprise']), axis=1)
    gdf['Aire_Polygone_Corrigée'] = 0.49 * gdf['Aire_Polygone'] ** 1.12
    gdf['Longueur_Emprise_Corrigée'] = 0.88 * gdf['Longueur_Emprise'] ** 1.02
    gdf['Largeur_Emprise_Corrigée'] = 0.48 * gdf['Largeur_Emprise'] ** 1.24
    gdf['Volume_Corrigé'] = 0.24 * gdf['Volume'] ** 1.17

    appliquer_filtre = input("Souhaitez-vous appliquer un filtre pour supprimer les polygones dont la longueur ou la largeur dépasse un seuil ? (oui/non): ")
    if appliquer_filtre.lower() == 'oui':
        seuil = float(input("Veuillez saisir la taille du seuil: "))
        gdf = gdf[(gdf['Longueur_Emprise_Corrigée'] <= seuil) & (gdf['Largeur_Emprise_Corrigée'] <= seuil)]
    gdf.drop(columns=['Aire_Emprise', 'Aire_Polygone', 'Longueur_Emprise', 'Largeur_Emprise', 'Volume'], inplace=True)
    gdf.rename(columns={
        'Aire_Polygone_Corrigée': 'Aire',
        'Longueur_Emprise_Corrigée': 'Longueur',
        'Largeur_Emprise_Corrigée': 'Largeur',
        'Volume_Corrigé': 'Volume'
    }, inplace=True)
    output_shapefile = shapefile_path.replace('.shp', '_metric.shp')
    gdf.to_file(output_shapefile)
    return output_shapefile


def main():
    parser = argparse.ArgumentParser(description="Process satellite images to extract and calculate various indices, classify, correct, vectorize and compute volume metrics.")
    parser.add_argument("input_tiff", help="Input TIFF file path.")
    parser.add_argument("model_path", help="Model path for classification.")
    parser.add_argument("keep_all_outputs", help=" to keep all outputs, 0 to delete them")

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

