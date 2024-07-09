#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from osgeo import gdal
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

def calculate_ndvi(input_tiff):
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
    otb_path = os.getenv('OTB_BIN_PATH')
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
    raster = rasterio.open(raster_path)
    band1 = raster.read(1)
    band2 = raster.read(2)
    band3 = raster.read(3)
    band4 = raster.read(4)
    band5 = raster.read(5)
    band6 = raster.read(6)
    band7 = raster.read(7)
    stacked_bands = np.stack((band1, band2, band3, band4, band5, band6, band7), axis=2)
    num_rows, num_cols, num_bands = stacked_bands.shape
    pixels = stacked_bands.reshape(num_rows * num_cols, num_bands)
    rf_optimized = joblib.load(model_path)
    predictions = rf_optimized.predict(pixels)
    predicted_image = predictions.reshape(num_rows, num_cols)

    def segment_components(image):
        all_labels = np.zeros_like(image, dtype=int)
        current_label = 1
        for value in range(1, 5):
            labeled_image, num_features = label(image == value)
            labeled_image[labeled_image > 0] += (current_label - 1)
            all_labels += labeled_image
            current_label += num_features
        return all_labels, current_label - 1

    labeled_image, num_features = segment_components(predicted_image)
    objects = find_objects(labeled_image)
    component_sizes = [np.sum(labeled_image == (i + 1)) for i in range(num_features)]
    corrected_image = predicted_image.copy()
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(num_features):
        coords = np.argwhere(labeled_image == (i + 1))
        if len(coords) < min_size:
            surrounding_values = []
            for y, x in coords:
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < predicted_image.shape[0] and 0 <= nx < predicted_image.shape[1]:
                        surrounding_values.append(predicted_image[ny, nx])
            if surrounding_values:
                most_frequent_value = Counter(surrounding_values).most_common(1)[0][0]
                for y, x in coords:
                    corrected_image[y, x] = most_frequent_value

    output_profile = raster.profile
    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress=Compression.deflate
    )

    with rasterio.open(output_corrected_path, 'w', **output_profile) as dst:
        dst.write(corrected_image.astype(rasterio.uint8), 1)

    src = rasterio.open(output_corrected_path)
    polygons = []
    for result in shapes(src.read(1), mask=src.read(1) == value_to_keep, connectivity=8):
        if Polygon(shape(result[0])).area >= min_size:
            polygons.append(shape(result['geometry']))

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)
    gdf.to_file(output_shapefile)
    print(f"Les polygones avec la valeur {value_to_keep} ont été sauvegardés dans {output_shapefile}")

def calculer_volume(shapefile_path):
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
    parser.add_argument("output_tiff_text", help="Output Texture TIFF file path.")
    parser.add_argument("output_tiff_merge", help="Merged indices TIFF file path.")
    parser.add_argument("model_path", help="Model path for classification.")
    parser.add_argument("output_corrected_path", help="Output corrected image path.")
    parser.add_argument("output_shapefile", help="Output shapefile path.")
    parser.add_argument("shapefile_path", help="Shapefile path for volume calculation.")
    args = parser.parse_args()
    ndvi = calculate_ndvi(args.input_tiff)
    brightness = calculate_brightness(args.input_tiff)
    calculate_texture(args.input_tiff, args.output_tiff_text)
    merge_indices_with_input(args.input_tiff, ndvi, brightness, args.output_tiff_text, args.output_tiff_merge)
    classify_correct_vectorize(args.output_tiff_merge, args.model_path, args.output_corrected_path, args.output_shapefile, value_to_keep=1, min_size=400)
    calculer_volume(args.shapefile_path)

if __name__ == "__main__":
    main()

