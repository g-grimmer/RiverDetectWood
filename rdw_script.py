#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install matplotlib')


# In[199]:


get_ipython().system('pip install xgboost')


# In[20]:


get_ipython().system('pip install optuna')


# In[20]:


get_ipython().system('pip install seaborn')


# In[76]:


get_ipython().system('pip install hyperopt')


# In[2]:


get_ipython().system('pip install scipy')


# In[3]:


get_ipython().system('pip install geopandas')


# In[4]:


get_ipython().system('pip install rasterio')


# In[3]:


get_ipython().system(' pip install scikit-learn')


# In[3]:


get_ipython().system('pip install scikit-image')


# In[80]:


get_ipython().system(' pip install rioxarray')


# In[1]:


import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import osgeo.gdal as gdal
import osgeo.ogr as ogr 
import osgeo.osr as osr
from osgeo.gdalconst import *
import fiona
import rasterio
from rasterio.features import shapes
import os
import subprocess
from osgeo import gdal, gdal_array
from rasterio.mask import mask
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from skimage import exposure
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from hyperopt import hp, fmin, tpe, Trials, space_eval
import random
from scipy.ndimage import grey_closing, generate_binary_structure
from hyperopt import hp, fmin, tpe, Trials, space_eval
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score
import csv
import optuna
from optuna.samplers import TPESampler



# In[99]:


import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
import geopandas as gpd

def vectorize_raster(raster_path, value_to_keep):
    # Ouvrir le fichier raster
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Lire la première bande du raster
        transform = src.transform
    
    # Créer un masque pour les pixels ayant la valeur spécifiée
    mask = image == value_to_keep
    
    # Vectoriser le raster
    results = (
        {'properties': {'value': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            shapes(image, mask=mask, transform=transform))
    )
    
    # Créer une liste de polygones shapely
    polygons = []
    for result in results:
        if result['properties']['value'] == value_to_keep:
            polygons.append(shape(result['geometry']))
    
    # Créer un GeoDataFrame à partir des polygones
    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)
    
    return gdf

# Chemin vers votre fichier raster
raster_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_10\meurthe_zoi4_band_ndvi_homred_brightness_entropred_predicted_corr_clip.tif"

# Vectoriser le raster et conserver les polygones avec la valeur 1
gdf = vectorize_raster(raster_path, value_to_keep)

# Sauvegarder les polygones dans un fichier shapefile
output_shapefile = r'D:\TER_GRIMMER\Methodologie\traitements\metrique_bm_model_10\meurthe\meurthe_zoi4_poly.shp'
gdf.to_file(output_shapefile)

print(f"Les polygones avec la valeur {value_to_keep} ont été sauvegardés dans {output_shapefile}")


# #####################################################################################################################################

# Partie 1 : Extraire les valeurs des pixels dans le NIR, Red et Green et ensuite en faire des histogrammes

# Création d'un mask sur les raster de la band PIR selon la couche vecteur

# In[44]:


# Charger les géométries où le champ "type" est égal à 10
geoms = []
with fiona.open(r"D:\TER_GRIMMER\Methodologie\traitements\python\chenal_actif\chenal_sans_berge\chenal_sans_berge\chenal_sans_berge.shp") as shapefile:
    for feature in shapefile:
        if feature['properties']['FID'] == 0:
            geoms.append(feature["geometry"])

# Ouvrir l'image raster
with rasterio.open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_9\meurthe_zoi2_band_ndvi_homred_brightness_predicted.tif") as src:
    # Lire seulement la première bande de l'image
    src_band = src.read(1)

    # Appliquer le masque sur la première bande de l'image
    out_image, out_transform = mask(src, geoms, crop=True)
    out_meta = src.meta.copy()

# Mettre à jour les métadonnées
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

# Écrire l'image masquée dans un fichier TIFF
with rasterio.open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_9\meurthe_zoi2_band_ndvi_homred_brightness_predicted_clip.tif", "w", **out_meta) as dest:
    dest.write(out_image)


# Vectorisation des raster mask

# In[80]:


dataset_mask = gdal.Open(r"D:\TER_GRIMMER\Methodologie\traitements\metriques_bm\meurthe\meurthe_zoi4_corr.tif")
dataset_mask


# In[81]:


band_pir_mask = dataset_mask.GetRasterBand(1)
band_pir_mask


# In[82]:


proj = dataset_mask.GetProjection()
shp_proj = osr.SpatialReference()
shp_proj.ImportFromWkt(proj)

output_file = r"D:\TER_GRIMMER\Methodologie\traitements\metriques_bm\meurthe\meurthe_zoi4_corr_vector.shp"
call_drive = ogr.GetDriverByName('ESRI Shapefile')
create_shp = call_drive.CreateDataSource(output_file)
shp_layer = create_shp.CreateLayer('autre_green_poly', srs = shp_proj)
new_field = ogr.FieldDefn(str('Green_Value'), ogr.OFTInteger)
shp_layer.CreateField(new_field)

gdal.Polygonize(band_pir_mask, None, shp_layer, 0, [], callback = None)
create_shp.Destroy()
dataset_mask = None


# Création histogramme à partir couche vecteur 

# In[46]:


#bm_10_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\PIR\bm_pir_10_poly.shp")
#bm_10_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\rouge\bm_red_10_poly.shp")
#bm_10_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\vert\bm_green_10_poly.shp")
#bm_20_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\PIR\bm_pir_20_poly.shp")
#bm_20_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\rouge\bm_red_20_poly.shp")
#bm_20_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM\vert\bm_green_20_poly.shp")
#eau_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\eau\PIR\eau_pir_poly.shp")
#eau_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\eau\rouge\eau_red_poly.shp")
#eau_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\eau\vert\eau_green_poly.shp")
#sable_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\gravier_sable\PIR\gravier_sableç_pir_poly.shp")
#sable_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\gravier_sable\rouge\gravier_sable_red_poly.shp")
#sable_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\gravier_sable\vert\gravier_sable_green_poly.shp")
#vege_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\vegetation\PIR\vege_pir_poly.shp")
#vege_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\vegetation\rouge\vege_red_poly.shp")
#vege_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\vegetation\vert\vege_green_poly.shp")
#bm_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM_niv1\PIR\bm_pir_poly.shp")
#bm_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM_niv1\rouge\bm_red_poly.shp")
#bm_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\BM_niv1\vert\bm_green_poly.shp")
#autre_pir = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\autre\PIR\autre_pir_poly.shp")
#autre_red = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\autre\rouge\autre_red_poly.shp")
#autre_green = gpd.read_file(r"D:\TER_GRIMMER\Methodologie\traitements\python\hist_value\autre\vert\autre_green_poly.shp")


# In[49]:


# Champ à utiliser pour l'histogramme
champ = "Green_Valu"

# Calculer la moyenne
moyenne = eau_green[champ].mean()

# Créer l'histogramme
plt.figure(figsize=(10, 6))
eau_green[champ].hist(bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(moyenne, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(moyenne))
plt.title("Histogram of the Green values for the 'Water' class")
plt.xlabel("Green values")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


# Calculer les moyennes
moyenne1 = bm_10_green[champ].mean()
moyenne2 = bm_20_green[champ].mean()

# Créer l'histogramme
plt.figure(figsize=(10, 6))

# Histogramme pour la première couche
bm_10_green[champ].hist(bins=20, color='skyblue', edgecolor='black', alpha=0.5, label='ODW')

# Histogramme pour la deuxième couche
bm_20_green[champ].hist(bins=20, color='orange', edgecolor='black', alpha=0.5, label='NDW')

# Ajouter les moyennes
plt.axvline(moyenne1, color='blue', linestyle='dashed', linewidth=1, label='ODW Green mean: {:.2f}'.format(moyenne1))
plt.axvline(moyenne2, color='red', linestyle='dashed', linewidth=1, label='NDW Green mean: {:.2f}'.format(moyenne2))

plt.title("Histogram of the Green values for the 'ODW' and 'NDW' classes")
plt.xlabel("Green values")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV en utilisant la colonne "CLASS" comme index
df = pd.read_csv(r'D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\sign_train.csv', sep=",", decimal=".", index_col="CLASS")

# Exclure la ligne "bm_niv1" du DataFrame
df = df.drop("bm_niv1", axis=0, errors="ignore")

# Supprimer les colonnes "NDVI" et "HOM"
df = df.drop(columns=["NDVI", "HOM"], errors="ignore")

# Réorganiser les colonnes pour mettre "GREEN" en premier et "PIR" en dernier
columns_order = ['GREEN'] + [col for col in df.columns if col not in ['GREEN', 'PIR']] + ['PIR']
df = df[columns_order]

# Remplacer les valeurs des classes
class_replacements = {1: 'DW', 2: 'Vegetation', 3: 'Sand/Gravels', 4: 'Water'}
df.index = df.index.map(class_replacements)

# Définition des catégories
categories = df.index.tolist()

# Définition des bandes spectrales (converties en chaînes de caractères)
bandes = list(map(str, df.columns.tolist()))

# Configuration du graphique
plt.figure(figsize=(10, 6))

# Liste des couleurs que vous souhaitez utiliser pour chaque catégorie
couleurs = ['rosybrown', 'greenyellow', 'gold', 'deepskyblue']

# Parcours des catégories
for i, categorie in enumerate(categories):
    plt.plot(bandes, df.loc[categorie], marker='o', label=categorie, color=couleurs[i], linewidth=2)  # Changer l'épaisseur ici

# Configuration des axes
plt.ylim(50, 155)
plt.yticks(range(50, 151, 10))
plt.xlabel("Spectral bands")
plt.ylabel("Reflectance")
plt.title("Average reflectance of classes for the training set")

# Légende
plt.legend()

# Affichage du graphique
plt.grid(True)
plt.show()


# #####################################################################################################################################

# 2ème partie : Calcul du NDVI

# In[29]:


def calculate_ndvi(input_tiff, output_tiff):
    with gdal.config_option("CHECK_DISK_FREE_SPACE", "NO"):
        # Open the input GeoTIFF file
        dataset = gdal.Open(input_tiff)
        
        # Get image dimensions
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        
        # Create output GeoTIFF file
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_tiff, xsize, ysize, 1, gdal.GDT_Float32)
        out_dataset.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset.SetProjection(dataset.GetProjection())
        
        # Define block size
        block_size = 1024  # Adjust this value based on your system's memory capacity
        
        for y in range(0, ysize, block_size):
            if y + block_size < ysize:
                y_block_size = block_size
            else:
                y_block_size = ysize - y
            
            for x in range(0, xsize, block_size):
                if x + block_size < xsize:
                    x_block_size = block_size
                else:
                    x_block_size = xsize - x

                # Read the NIR and Red bands for the current block
                band_nir = dataset.GetRasterBand(1).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                band_red = dataset.GetRasterBand(2).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                
                # Calculate NDVI for the current block
                ndvi = (band_nir - band_red) / (band_nir + band_red + 1e-10)  # Adding a small constant to avoid division by zero
                
                # Write NDVI band
                out_band = out_dataset.GetRasterBand(1)
                out_band.WriteArray(ndvi, x, y)
        
        # Set NoData value
        out_band.SetNoDataValue(-9999)

        # Flush cache to ensure all data is written to disk
        out_band.FlushCache()

        # Close datasets
        dataset = None
        out_dataset = None

# Example usage
input_tiff = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis.tif"
output_tiff = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_ndvi.tif"
calculate_ndvi(input_tiff, output_tiff)


# Merge du raster chenal_actif et le ndvi

# In[75]:


with gdal.config_option("CHECK_DISK_FREE_SPACE", "NO"):
# Charger les rasters
    raster1 = gdal.Open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_band_ndvi_homred_brightness.tif")
    raster2 = gdal.Open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_texture.tif")

# Lire les bandes du premier raster (Int16, 3 bandes)
    band1_1 = raster1.GetRasterBand(1).ReadAsArray().astype(np.float32)
    band1_2 = raster1.GetRasterBand(2).ReadAsArray().astype(np.float32)
    band1_3 = raster1.GetRasterBand(3).ReadAsArray().astype(np.float32)
    band1_4 = raster1.GetRasterBand(4).ReadAsArray().astype(np.float32)
    band1_5 = raster1.GetRasterBand(5).ReadAsArray().astype(np.float32)
    band1_6 = raster1.GetRasterBand(6).ReadAsArray().astype(np.float32)

# Lire la bande du deuxième raster (Float32, 1 bande)
    band2_1 = raster2.GetRasterBand(2).ReadAsArray().astype(np.float32)

# Créer un nouveau raster avec 4 bandes et une compression élevée
    driver = gdal.GetDriverByName('GTiff')
    options = ['COMPRESS=LZW']
    out_raster = driver.Create(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_band_ndvi_homred_brightness_entropred.tif", 
                               raster1.RasterXSize, raster1.RasterYSize, 7, gdal.GDT_Float32, options=options)

# Copier les géoréférencements et les projections
    out_raster.SetGeoTransform(raster1.GetGeoTransform())
    out_raster.SetProjection(raster1.GetProjection())

# Écrire les 3 bandes du premier raster
    out_raster.GetRasterBand(1).WriteArray(band1_1.astype(np.float32))
    out_raster.GetRasterBand(2).WriteArray(band1_2.astype(np.float32))
    out_raster.GetRasterBand(3).WriteArray(band1_3.astype(np.float32))
    out_raster.GetRasterBand(4).WriteArray(band1_4.astype(np.float32))
    out_raster.GetRasterBand(5).WriteArray(band1_5.astype(np.float32))
    out_raster.GetRasterBand(6).WriteArray(band1_6.astype(np.float32))

# Écrire la bande du deuxième raster
    out_raster.GetRasterBand(7).WriteArray(band2_1)

# Fermer les rasters
    out_raster.FlushCache()
    del out_raster, raster1, raster2


# #####################################################################################################################################

# 3ème partie : Calcul de l'indice de brillance

# In[37]:


def calculate_ndvi(input_tiff, output_tiff):
    with gdal.config_option("CHECK_DISK_FREE_SPACE", "NO"):
        # Open the input GeoTIFF file
        dataset = gdal.Open(input_tiff)
        
        # Get image dimensions
        xsize = dataset.RasterXSize
        ysize = dataset.RasterYSize
        
        # Create output GeoTIFF file
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_tiff, xsize, ysize, 1, gdal.GDT_Float32)
        out_dataset.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset.SetProjection(dataset.GetProjection())
        
        # Define block size
        block_size = 1024  # Adjust this value based on your system's memory capacity
        
        for y in range(0, ysize, block_size):
            if y + block_size < ysize:
                y_block_size = block_size
            else:
                y_block_size = ysize - y
            
            for x in range(0, xsize, block_size):
                if x + block_size < xsize:
                    x_block_size = block_size
                else:
                    x_block_size = xsize - x

                # Read the NIR and Red bands for the current block
                band_nir = dataset.GetRasterBand(1).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                band_red = dataset.GetRasterBand(2).ReadAsArray(x, y, x_block_size, y_block_size).astype(np.float32)
                
                # Calculate NDVI for the current block
                brightness_index = np.sqrt(band_red**2 + band_nir**2)  # Adding a small constant to avoid division by zero
                
                # Write NDVI band
                out_band = out_dataset.GetRasterBand(1)
                out_band.WriteArray(brightness_index, x, y)
        
        # Set NoData value
        out_band.SetNoDataValue(-9999)

        # Flush cache to ensure all data is written to disk
        out_band.FlushCache()

        # Close datasets
        dataset = None
        out_dataset = None

# Example usage
input_tiff = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis.tif"
output_tiff = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_brightness.tif"
calculate_ndvi(input_tiff, output_tiff)


# Merge du raster chenal et indice brillance

# In[17]:


import numpy as np
from osgeo import gdal

# Configuration de GDAL pour ignorer l'espace disque libre
gdal.SetConfigOption("CHECK_DISK_FREE_SPACE", "NO")

# Charger les rasters
raster1 = gdal.Open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_9\meurthe_zoi4_band_ndvi_homred_brightness.tif")
raster2 = gdal.Open(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_5\meurthe_zoi4_texture.tif")

# Créer un nouveau raster avec 6 bandes et une compression élevée
driver = gdal.GetDriverByName('GTiff')
options = ['COMPRESS=LZW']
out_raster = driver.Create(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\meurthe_classif\classif_10\meurthe_zoi4_band_ndvi_homred_brightness_entropred.tif", 
                           raster1.RasterXSize, raster1.RasterYSize, 7, gdal.GDT_Float32, options=options)

# Copier les géoréférencements et les projections
out_raster.SetGeoTransform(raster1.GetGeoTransform())
out_raster.SetProjection(raster1.GetProjection())

# Définir la taille des blocs
block_size = 512

# Lire et écrire les rasters par blocs
for i in range(0, raster1.RasterYSize, block_size):
    if i + block_size < raster1.RasterYSize:
        num_rows = block_size
    else:
        num_rows = raster1.RasterYSize - i
    
    for j in range(0, raster1.RasterXSize, block_size):
        if j + block_size < raster1.RasterXSize:
            num_cols = block_size
        else:
            num_cols = raster1.RasterXSize - j
        
        # Lire les bandes du premier raster par blocs
        band1_1 = raster1.GetRasterBand(1).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        band1_2 = raster1.GetRasterBand(2).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        band1_3 = raster1.GetRasterBand(3).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        band1_4 = raster1.GetRasterBand(4).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        band1_5 = raster1.GetRasterBand(5).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        band1_6 = raster1.GetRasterBand(6).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        
        # Lire la bande du deuxième raster par blocs
        band2_1 = raster2.GetRasterBand(2).ReadAsArray(j, i, num_cols, num_rows).astype(np.float32)
        
        # Écrire les blocs dans le nouveau raster
        out_raster.GetRasterBand(1).WriteArray(band1_1, j, i)
        out_raster.GetRasterBand(2).WriteArray(band1_2, j, i)
        out_raster.GetRasterBand(3).WriteArray(band1_3, j, i)
        out_raster.GetRasterBand(4).WriteArray(band1_4, j, i)
        out_raster.GetRasterBand(5).WriteArray(band1_5, j, i)
        out_raster.GetRasterBand(6).WriteArray(band1_6, j, i)
        out_raster.GetRasterBand(7).WriteArray(band2_1, j, i)

# Fermer les rasters
out_raster.FlushCache()
del out_raster, raster1, raster2


# #####################################################################################################################################

# 3ème partie, calcul de l'homogénéité sur la base du RED

# In[23]:


import subprocess

# Définir les arguments de la commande
command = [
    r"D:\TER_GRIMMER\OTB\bin\otbcli_HaralickTextureExtraction.bat",
    "-in", r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\zoi4.tif",
    "-channel", "2",
    "-parameters.xrad", "3",
    "-parameters.yrad", "3",
    "-texture", "simple",
    "-out", r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\zoi4_red.tif"
]

# Exécuter la commande
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("Commande exécutée avec succès")
    print("Sortie standard:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Erreur lors de l'exécution de la commande")
    print("Code de retour:", e.returncode)
    print("Erreur standard:", e.stderr)


# #####################################################################################################################################

# 4ème partie : Classification supervisée par modèle RF

# Extraction des données

# In[55]:


from rasterio.mask import mask


# In[5]:


# Charger le shapefile
shapefile_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\roi_bm1_test.shp"
gdf = gpd.read_file(shapefile_path)

# Charger l'image raster
raster_path = r"D:\TER_GRIMMER\Methodologie\traitements\python\chenal_actif\chenal_sans_berge\meurthe\chenal_test_band_ndvi_homred_brightness_entropred.tif"
raster = rasterio.open(raster_path)

# Initialiser une liste pour stocker les données extraites
data = []

# Parcourir chaque polygone
for _, row in gdf.iterrows():
    geom = [row['geometry']]
    class_code = row['type_niv1']
    
    # Masquer le raster avec le polygone actuel
    out_image, out_transform = mask(raster, geom, crop=True)
    
    # Obtenir les valeurs des pixels pour chaque bande
    PIR_values = out_image[0].flatten()  # Première bande
    RED_values = out_image[1].flatten()  # Deuxième bande
    GREEN_values = out_image[2].flatten()  # Troisième bande
    NDVI_values = out_image[3].flatten()  # Troisième bande
    HOM_values = out_image[4].flatten()  # Troisième bande
    BRIGHT_values = out_image[5].flatten()  # Troisième bande
    ENT_values = out_image[6].flatten()  # Troisième bande
    
    # Filtrer les valeurs masquées (valeurs égales à nodata)
    valid_mask = (PIR_values != raster.nodata) & (RED_values != raster.nodata) & (GREEN_values != raster.nodata) & (NDVI_values != raster.nodata) & (HOM_values != raster.nodata) & (BRIGHT_values != raster.nodata)  & (ENT_values != raster.nodata)
     
    PIR_values = PIR_values[valid_mask]
    RED_values = RED_values[valid_mask]
    GREEN_values = GREEN_values[valid_mask]
    NDVI_values = NDVI_values[valid_mask]
    HOM_values = HOM_values[valid_mask]
    BRIGHT_values = BRIGHT_values[valid_mask]
    ENT_values = ENT_values[valid_mask]

    # Ajouter les valeurs et le code de classe dans la liste de données
    for pir, red, green, ndvi, hom, bright, ent in zip(PIR_values, RED_values, GREEN_values, NDVI_values, HOM_values, BRIGHT_values, ENT_values):
        data.append([pir, red, green, ndvi, hom, bright, ent, class_code])

# Créer un DataFrame avec les données extraites
df = pd.DataFrame(data, columns=['PIR', 'RED', 'GREEN', 'NDVI', 'HOM','BRIGHT','ENT', 'CLASS'])

# Sauvegarder le DataFrame en fichier CSV
output_csv_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\output_test.csv"
df.to_csv(output_csv_path, index=False)

print(f"Les données ont été extraites et enregistrées dans {output_csv_path}")


# Nettoyage et correction du fichier CSV

# In[6]:


# Chemin vers le fichier CSV
input_csv_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\output_test.csv"
output_csv_cleaned_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\output_test_clean.csv"

# Charger le fichier CSV
df = pd.read_csv(input_csv_path)

# Supprimer les lignes contiennent la valeur 0
df_cleaned = df[(df['PIR'] != 0) & (df['RED'] != 0) & (df['GREEN'] != 0) & (df['NDVI'] != 0) & (df['HOM'] != 0) & (df['BRIGHT'] != 0) & (df['ENT'] != 0)]

# Supprimer les lignes où la colonne CLASS contient la valeur 5
df_cleaned = df_cleaned[df_cleaned['CLASS'] != 5]

# Sauvegarder le DataFrame nettoyé en fichier CSV
df_cleaned.to_csv(output_csv_cleaned_path, index=False)

print(f"Les données nettoyées ont été enregistrées dans {output_csv_cleaned_path}")


# Application du modèle RF

# Méthode par séparation du chenal

# In[7]:


import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Lire les fichiers CSV pour les données d'entraînement et de test
df_train = pd.read_csv(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\output_train_clean.csv")
df_test = pd.read_csv(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\output_test_clean.csv")

# Vérifier les colonnes des dataframes
print("Colonnes du dataframe d'entraînement :")
print(df_train.columns)
print("Colonnes du dataframe de test :")
print(df_test.columns)

# Séparer les features et la cible pour les données d'entraînement
X_train = df_train[['PIR', 'RED', 'GREEN', 'NDVI', 'HOM', 'BRIGHT', 'ENT']]
y_train = df_train['CLASS']

# Séparer les features et la cible pour les données de test
X_test = df_test[['PIR', 'RED', 'GREEN', 'NDVI', 'HOM', 'BRIGHT', 'ENT']]
y_test = df_test['CLASS']

# Vérifier la distribution des classes avant équilibrage dans les données d'entraînement
print("Distribution des classes avant équilibrage dans les données d'entraînement :")
print(y_train.value_counts())

# Équilibrer les classes dans les données d'entraînement
min_class_count_train = y_train.value_counts().min()
dfs_train = []
for cls in y_train.unique():
    class_df = df_train[df_train['CLASS'] == cls]
    sampled_class_df = resample(class_df, 
                                replace=False, # échantillonnage sans remplacement
                                n_samples=min_class_count_train, 
                                random_state=42) # fixe la graine pour la reproductibilité
    dfs_train.append(sampled_class_df)

balanced_train_df = pd.concat(dfs_train)

# Vérifier la distribution des classes après équilibrage dans les données d'entraînement
print("Distribution des classes après équilibrage dans les données d'entraînement :")
print(balanced_train_df['CLASS'].value_counts())

# Séparer les features et la cible du dataframe équilibré pour l'entraînement
X_balanced_train = balanced_train_df[['PIR', 'RED', 'GREEN', 'NDVI', 'HOM', 'BRIGHT', 'ENT']]
y_balanced_train = balanced_train_df['CLASS']

# Initialiser et entraîner le modèle de forêt aléatoire avec les meilleurs hyperparamètres
rf_optimized = RandomForestClassifier(n_estimators = 967, max_features = 1, max_depth = 18, min_samples_split = 10, min_samples_leaf = 18, random_state=42)
rf_optimized.fit(X_balanced_train, y_balanced_train)

# Prédire les labels pour l'ensemble de test avec le modèle entraîné
y_pred = rf_optimized.predict(X_test)

# Enregistrer le modèle dans un fichier .rf
joblib.dump(rf_optimized, r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\classif_model_10.rf")

# Vérifier la distribution des prédictions
print("Distribution des prédictions :")
print(pd.Series(y_pred).value_counts())

# Évaluer le modèle
accuracy = rf_optimized.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Générer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion sous forme de heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_train.unique(), yticklabels=y_train.unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))


# Méthode avec application de l'algo TPE

# In[21]:


# Lire le fichier CSV
df = pd.read_csv(r"D:\TER_GRIMMER\Methodologie\traitements\classif\output_data_chenal_clean.csv")

# Vérifier les colonnes du dataframe
print(df.columns)

# Séparer les features et la cible
X = df[['PIR', 'RED', 'GREEN', 'NDVI']]
y = df['CLASS']

# Vérifier la distribution des classes avant équilibrage
print("Distribution des classes avant équilibrage :")
print(y.value_counts())

# Équilibrer les classes
min_class_count = y.value_counts().min()
dfs = []
for cls in y.unique():
    class_df = df[df['CLASS'] == cls]
    sampled_class_df = resample(class_df, 
                                replace=False, # échantillonnage sans remplacement
                                n_samples=min_class_count, 
                                random_state=42) # fixe la graine pour la reproductibilité
    dfs.append(sampled_class_df)

balanced_df = pd.concat(dfs)

# Vérifier la distribution des classes après équilibrage
print("Distribution des classes après équilibrage :")
print(balanced_df['CLASS'].value_counts())

# Séparer les features et la cible du dataframe équilibré
X_balanced = balanced_df[['PIR', 'RED', 'GREEN', 'NDVI']]
y_balanced = balanced_df['CLASS']

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Définir la fonction objectif pour Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 1000)
    max_features = trial.suggest_int('max_features', 1, 20)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy').mean()
    return score

# Créer une étude et optimiser les hyperparamètres
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=15)

# Afficher les meilleurs hyperparamètres
best_params = study.best_params
print("Best hyperparameters found:")
print(best_params)

# Initialiser et entraîner le modèle de forêt aléatoire avec les meilleurs hyperparamètres
rf_optimized = RandomForestClassifier(**best_params, random_state=42)
rf_optimized.fit(X_train, y_train)

# Utiliser la méthode Bootstrap pour évaluer le modèle
n_iterations = 100
bootstrap_scores = []

for i in range(n_iterations):
    # Créer un échantillon bootstrap
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, random_state=i)
    # Entraîner le modèle sur l'échantillon bootstrap
    rf_optimized.fit(X_train_bootstrap, y_train_bootstrap)
    # Évaluer le modèle sur le jeu de test
    bootstrap_score = rf_optimized.score(X_test, y_test)
    bootstrap_scores.append(bootstrap_score)

# Afficher les résultats de l'évaluation Bootstrap
bootstrap_scores = np.array(bootstrap_scores)
print(f'Bootstrap Accuracy: {bootstrap_scores.mean():.2f} +/- {bootstrap_scores.std():.2f}')

# Prédire les labels pour l'ensemble de test avec le modèle entraîné initialement
y_pred = rf_optimized.predict(X_test)

# Vérifier la distribution des prédictions
print("Distribution des prédictions :")
print(pd.Series(y_pred).value_counts())

# Évaluer le modèle
accuracy = rf_optimized.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Effectuer une validation croisée en 5 volets
cv_results = cross_validate(rf_optimized, X_balanced, y_balanced, cv=5, scoring='accuracy', return_train_score=True)

# Afficher les résultats de la validation croisée
print("Résultats de la validation croisée (5 volets) :")
print(f"Train accuracy : {cv_results['train_score'].mean():.2f} +/- {cv_results['train_score'].std():.2f}")
print(f"Test accuracy  : {cv_results['test_score'].mean():.2f} +/- {cv_results['test_score'].std():.2f}")

# Créer un DataFrame à partir des résultats de la validation croisée
cv_df = pd.DataFrame(cv_results)
cv_df['Fold'] = cv_df.index + 1  # Ajouter une colonne pour les plis

# Plot de la cross validation avec Seaborn
plt.figure(figsize=(10, 5))
sns.lineplot(data=cv_df, x='Fold', y='train_score', label='Train Accuracy')
sns.lineplot(data=cv_df, x='Fold', y='test_score', label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross Validation Scores')
plt.legend()
plt.show()

# Générer la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)

# Afficher la matrice de confusion sous forme de heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))


# Application du modèle sur l'image voulue

# In[79]:


import numpy as np
import rasterio
from rasterio.enums import Compression

# Charger l'image raster
raster_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\indices\buech_zoi4_bis_band_ndvi_homred_brightness_entropred.tif"
raster = rasterio.open(raster_path)

# Lire les bandes
band1 = raster.read(1)  # PIR
band2 = raster.read(2)  # RED
band3 = raster.read(3)  # GREEN
band4 = raster.read(4)  # NDVI
band5 = raster.read(5) # HOM
band6 = raster.read(6) # BRIGHT
band7 = raster.read(7) # ENT

# Empiler les bandes pour obtenir une matrice 3D
stacked_bands = np.stack((band1, band2, band3, band4, band5, band6, band7), axis=2)

# Reshaper la matrice en 2D où chaque ligne est un pixel et les colonnes sont les bandes
num_rows, num_cols, num_bands = stacked_bands.shape
pixels = stacked_bands.reshape(num_rows * num_cols, num_bands)

# Charger le modèle de forêt aléatoire préalablement entraîné
# Assurez-vous que le modèle est déjà entraîné et disponible dans la variable `rf_optimized`
rf_optimized = joblib.load(r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\classif_model_10.rf")  # Charger le modèle sauvegardé

# Prédire les classes pour chaque pixel
predictions = rf_optimized.predict(pixels)

# Reshaper les prédictions en 2D pour correspondre à l'image raster
predicted_image = predictions.reshape(num_rows, num_cols)

# Définir le profil de l'image de sortie
output_profile = raster.profile
output_profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress=Compression.deflate  # Utiliser une compression compatible, par exemple 'deflate'
)

# Enregistrer l'image prédite en tant que nouveau fichier raster
output_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\buech_zoi4_bis_band_ndvi_homred_brightness_entropred_predicted.tif"
with rasterio.open(output_path, 'w', **output_profile) as dst:
    dst.write(predicted_image.astype(rasterio.uint8), 1)

print("L'image prédite a été enregistrée avec succès.")


# Composantes connexes

# In[83]:


import numpy as np
from scipy.ndimage import label, find_objects
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from collections import Counter

# Charger l'image raster GeoTIFF
input_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\buech_zoi4_bis_band_ndvi_homred_brightness_entropred_predicted.tif"
output_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\buech_classif\classif_10_bis\buech_zoi4_bis_band_ndvi_homred_brightness_entropred_predicted_corr.tif"

with rasterio.open(input_path) as src:
    image = src.read(1)  # Lire la première bande

# Fonction pour segmenter les composantes connexes pour chaque valeur de pixel
def segment_components(image):
    all_labels = np.zeros_like(image, dtype=int)
    current_label = 1
    for value in range(1, 5):  # Les valeurs de pixels vont de 1 à 4
        labeled_image, num_features = label(image == value)
        labeled_image[labeled_image > 0] += (current_label - 1)
        all_labels += labeled_image
        current_label += num_features
    return all_labels, current_label - 1

# Segmenter les composantes connexes
labeled_image, num_features = segment_components(image)

# Trouver les objets (composantes connexes)
objects = find_objects(labeled_image)

# Afficher les tailles des composantes connexes pour débogage
component_sizes = [np.sum(labeled_image == (i + 1)) for i in range(num_features)]

# Filtrer et corriger les petites composantes
min_size = 400  # Taille minimale pour ne pas être considéré comme bruit
corrected_image = image.copy()

# Coordonnées des 8 voisins (8-adjacence)
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

for i in range(num_features):
    # Récupérer les pixels de la composante connexe
    coords = np.argwhere(labeled_image == (i + 1))
    
    # Si la taille de la composante est inférieure à min_size, on la corrige
    if len(coords) < min_size:
        # Trouver les valeurs des pixels environnants
        surrounding_values = []
        for y, x in coords:
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    surrounding_values.append(image[ny, nx])
        
        # Remplacer par la valeur la plus fréquente des pixels environnants
        if surrounding_values:
            most_frequent_value = Counter(surrounding_values).most_common(1)[0][0]
            for y, x in coords:
                corrected_image[y, x] = most_frequent_value

# Enregistrer l'image corrigée dans un nouveau fichier GeoTIFF
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=corrected_image.shape[0],
    width=corrected_image.shape[1],
    count=1,
    dtype=corrected_image.dtype,
    crs=src.crs,
    transform=src.transform,
) as dst:
    dst.write(corrected_image, 1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


import argparse
import numpy as np
from osgeo import gdal
import subprocess
import numpy as np
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
    command = [
        r"D:\TER_GRIMMER\OTB\bin\otbcli_HaralickTextureExtraction.bat",
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

    print("L'image corrigée a été enregistrée avec succès.")

    with rasterio.open(output_corrected_path) as src:
        image = src.read(1)
        transform = src.transform

    mask = image == value_to_keep
    results = (
        {'properties': {'value': v}, 'geometry': s}
        for i, (s, v) in enumerate(
            shapes(image, mask=mask, transform=transform))
    )

    polygons = []
    for result in results:
        if result['properties']['value'] == value_to_keep:
            polygons.append(shape(result['geometry']))

    gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=src.crs)
    gdf.to_file(output_shapefile)

    print(f"Les polygones avec la valeur {value_to_keep} ont été sauvegardés dans {output_shapefile}")


def calculer_volume(shapefile_path):
    # Charger le shapefile en tant que GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # Fonction pour calculer l'aire d'un polygone
    def calculer_aire(poly):
        return poly.area

    # Fonction pour calculer l'emprise minimum orientée (minimum rotated bounding box)
    def calculer_emprise_minimum(poly):
        # Calculer l'enveloppe convexe
        convex_hull = poly.convex_hull

        # Calculer l'emprise minimum orientée (minimum rotated rectangle)
        min_rect = convex_hull.minimum_rotated_rectangle

        # Calculer la longueur et la largeur du rectangle
        bounds = min_rect.bounds
        longueur = bounds[2] - bounds[0]
        largeur = bounds[3] - bounds[1]

        # Calculer l'aire de l'emprise minimum
        aire_emprise = min_rect.area

        return longueur, largeur, aire_emprise

    # Fonction pour calculer le volume
    def calculer_volume_polygone(poly, aire_polygone, longueur_emprise, largeur_emprise, aire_emprise):
        # Calculer le facteur de correction c
        c = aire_polygone / aire_emprise

        # Calculer le volume
        volume = c * longueur_emprise * (largeur_emprise ** 2) * (math.pi / 4)

        return volume

    # Ajouter les colonnes pour stocker les résultats
    gdf['Aire_Polygone'] = gdf.geometry.apply(calculer_aire)
    gdf['Longueur_Emprise'], gdf['Largeur_Emprise'], gdf['Aire_Emprise'] = zip(*gdf.geometry.apply(calculer_emprise_minimum))

    # Calculer le volume pour chaque polygone
    gdf['Volume'] = gdf.apply(lambda row: calculer_volume_polygone(row.geometry, row['Aire_Polygone'], row['Longueur_Emprise'], row['Largeur_Emprise'], row['Aire_Emprise']), axis=1)

    # Appliquer les corrections aux valeurs calculées
    gdf['Aire_Polygone_Corrigée'] = 0.49 * gdf['Aire_Polygone'] ** 1.12
    gdf['Longueur_Emprise_Corrigée'] = 0.88 * gdf['Longueur_Emprise'] ** 1.02
    gdf['Largeur_Emprise_Corrigée'] = 0.48 * gdf['Largeur_Emprise'] ** 1.24
    gdf['Volume_Corrigé'] = 0.24 * gdf['Volume'] ** 1.17

    # Demander à l'utilisateur s'il souhaite appliquer un filtre
    appliquer_filtre = input("Souhaitez-vous appliquer un filtre pour supprimer les polygones dont la longueur ou la largeur dépasse un seuil ? (oui/non): ")

    if appliquer_filtre.lower() == 'oui':
        # Demander à l'utilisateur de saisir la taille du seuil
        seuil = float(input("Veuillez saisir la taille du seuil: "))
        
        # Appliquer le filtre
        gdf = gdf[(gdf['Longueur_Emprise_Corrigée'] <= seuil) & (gdf['Largeur_Emprise_Corrigée'] <= seuil)]

    # Supprimer les colonnes non corrigées du GeoDataFrame si nécessaire
    gdf.drop(columns=['Aire_Emprise', 'Aire_Polygone', 'Longueur_Emprise', 'Largeur_Emprise', 'Volume'], inplace=True)

    # Renommer les colonnes corrigées si nécessaire
    gdf.rename(columns={
        'Aire_Polygone_Corrigée': 'Aire',
        'Longueur_Emprise_Corrigée': 'Longueur',
        'Largeur_Emprise_Corrigée': 'Largeur',
        'Volume_Corrigé': 'Volume'
    }, inplace=True)

    # Créer une nouvelle couche shapefile avec les résultats
    output_shapefile = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_poly_metric.shp"
    gdf.to_file(output_shapefile)

    return output_shapefile

input_tiff = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\m_zoi2_chenal.tif" #Input Image ROI
output_tiff_text = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_text.tif" #Output Texture
output_tiff_merge = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_merge.tif" #Merge des indices
model_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\typo_wtht_others\classif_10_band_ndvi_homred_brightness_entropyred\classif_model_10.rf" #Lien vers le modèle
output_corrected_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_classif.tif" #Lien image corrigée
output_shapefile = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_poly.shp" #Lien sortie vecteur
shapefile_path = r"D:\TER_GRIMMER\Methodologie\traitements\classif\app_distr_homo\classif_outil\meurthe_zoi4_poly.shp"


ndvi = calculate_ndvi(input_tiff)
brightness = calculate_brightness(input_tiff)
calculate_texture(input_tiff, output_tiff_text)
merge_indices_with_input(input_tiff, ndvi, brightness, output_tiff_text, output_tiff_merge)
classify_correct_vectorize(output_tiff_merge, model_path, output_corrected_path, output_shapefile, value_to_keep=1, min_size=400)
calculer_volume(shapefile_path)


# In[ ]:




