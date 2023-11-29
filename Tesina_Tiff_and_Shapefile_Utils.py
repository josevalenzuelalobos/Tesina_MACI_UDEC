import shutil  
import numpy as np
import rasterio
import geopandas as gpd
import os
from rasterio.features import geometry_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import defaultdict
import numpy as np

def extract_pixels_within_mask(tiff_path, mask):
    """
    Extracts pixels from a TIFF file within the specified mask areas. Returns a list of pixel values from multiple bands and their respective band names.
    :param tiff_path: Path to the TIFF file.
    :param mask: Binary mask to apply to the TIFF.
    :return: Tuple of (masked pixel values, band names).
    """
    with rasterio.open(tiff_path) as src:
        band_names = src.descriptions
        band_names = [f'band_{i+1}' if name is None else name for i, name in enumerate(band_names)]
        masked_pixels = [src.read(bidx)[mask] for bidx in range(1, src.count+1)]
        masked_pixels = np.array(masked_pixels).T.tolist()
    return masked_pixels, band_names


def process_shapefile_data(directory_path, base_path_to_remove=None):
    """
    Processes shapefiles in a given directory and extracts related metadata. Optionally, removes a base path from TIFF file paths.
    :param directory_path: Directory containing shapefiles.
    :param base_path_to_remove: Base path to remove from TIFF file paths (if any).
    :return: List of metadata dictionaries for each shapefile.
    """
    metadata_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".shp"):
                gdf = gpd.read_file(os.path.join(root, file))
                first_row = gdf.iloc[0]
                tif_file, prev_tif_file = map(lambda x: x.replace(base_path_to_remove, '') if base_path_to_remove else x, (first_row['tif'], first_row['prev_tif']))
                raster_shape, transform, crs = get_raster_info(tif_file)
                mask = geometry_mask(gdf.geometry.values, transform=transform, invert=True, out_shape=raster_shape)
                metadata = {key: first_row[key] for key in ['zone', 'region', 'type_index', 'label']}
                metadata.update({'tif': tif_file, 'prev_tif': prev_tif_file, 'height': raster_shape[0], 'width': raster_shape[1], 'mask': mask})
                metadata_list.append(metadata)
    return metadata_list


def get_raster_info(tiff_path):
    """
    Retrieves basic information from a TIFF file, such as shape, transform, and CRS.
    :param tiff_path: Path to the TIFF file.
    :return: Tuple of (raster shape, transform, CRS).
    """
    with rasterio.open(tiff_path) as src:
        return src.shape, src.transform, src.crs


def reproject_tiff_files(input_dir, output_dir, target_crs='EPSG:4326'):
    """
    Reprojects TIFF files from an input directory to a specified CRS, saving them in an output directory.
    :param input_dir: Directory containing input TIFF files.
    :param output_dir: Directory to save the reprojected TIFF files.
    :param target_crs: CRS to reproject to (default: 'EPSG:4326').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.endswith(('.tif', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            with rasterio.open(input_path) as src:
                transform, width, height = calculate_default_transform(src.crs, target_crs, src.width, src.height, *src.bounds)
                metadata = src.meta.copy()
                metadata.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
                with rasterio.open(output_path, 'w', **metadata) as dst:
                    for i in range(1, src.count + 1):
                        reproject(source=rasterio.band(src, i), destination=rasterio.band(dst, i), src_transform=src.transform, src_crs=src.crs, dst_transform=transform, dst_crs=target_crs, resampling=Resampling.nearest)
                        dst.update_tags(i, src.tags(i))
                        if src.descriptions and src.descriptions[i-1]:
                            dst.set_band_description(i, src.descriptions[i-1])
            copy_auxiliary_files(input_path, output_path)

def copy_auxiliary_files(input_path, output_path):
    """
    Copies auxiliary files related to a TIFF file, such as '.aux.xml', from the input path to the output path.
    :param input_path: Path to the input TIFF file.
    :param output_path: Path to the output TIFF file.
    """
    aux_xml_path = input_path + '.aux.xml'
    if os.path.exists(aux_xml_path):
        shutil.copy(aux_xml_path, output_path + '.aux.xml')

def analyze_masks_and_labels(shapefile_data):
    """
    Analyzes the masks and labels in shapefile data, counting occurrences and true/false values within masks for each label.
    :param shapefile_data: List of dictionaries containing shapefile metadata, including masks and labels.
    """
    true_counts, false_counts, occurrences = defaultdict(int), defaultdict(int), defaultdict(int)
    for entry in shapefile_data:
        mask = entry.get('mask')
        label = entry.get('label')
        if mask is not None and label is not None:
            true_counts[label] += np.sum(mask)
            false_counts[label] += np.size(mask) - np.sum(mask)
            occurrences[label] += 1
    # Printing results
    print_occurrences_and_mask_values(occurrences, true_counts, false_counts)

def print_occurrences_and_mask_values(occurrences, true_counts, false_counts):
    """
    Prints the occurrences and true/false counts of labels in masks.
    :param occurrences: Dictionary of label occurrences.
    :param true_counts: Dictionary of true value counts per label.
    :param false_counts: Dictionary of false value counts per label.
    """
    print("Label occurrences:")
    for label, count in occurrences.items():
        print(f"{label}: {count} times")
    print("\nTrue values in masks by label:")
    for label, count in true_counts.items():
        print(f"{label}: {count} True values")
    print("\nFalse values in masks by label:")
    for label, count in false_counts.items():
        print(f"{label}: {count} False values")

def enrich_metadata_with_pixel_data(metadata_list):
    """
    Enriches shapefile metadata with pixel data and band names from corresponding TIFF files.
    :param metadata_list: List of metadata dictionaries for each shapefile.
    :return: Updated list of metadata dictionaries.
    """
    for entry in metadata_list:
        tiff_file, prev_tiff_file, mask = entry['tif'], entry['prev_tif'], entry['mask']
        pixels, band_names = extract_pixels_within_mask(tiff_file, mask)
        entry.update({'pixels': pixels, 'band_names': band_names, 'prev_pixels': extract_pixels_within_mask(prev_tiff_file, mask)[0]})
    return metadata_list


def reorder_and_select_specific_bands(output_data, required_bands):
    """
    Reorders and selects specific bands of pixel data in each element of output_data based on required_bands.
    Raises an exception if any element is missing any of the required bands.

    :param output_data: List of dictionaries, each containing 'band_names' and 'pixels'.
    :param required_bands: List of band names to reorder and select.
    :return: The modified output_data with pixels reordered and filtered according to required_bands.
    """
    for entry in output_data:
        band_names = entry['band_names']
        pixels = entry['pixels']

        if not all(band in band_names for band in required_bands):
            raise Exception(f"Missing required bands in the element: {band_names}")

        band_index_map = {band: band_names.index(band) for band in required_bands}
        reordered_pixels = [[pixel[band_index_map[band]] for band in required_bands] for pixel in pixels]

        entry['pixels'] = reordered_pixels
        entry['band_names'] = required_bands

    return output_data


def normalize_pixels_globally_with_margin(output_data, margin_factor=0.10):
    """
    Normalizes pixel values globally across all entries in output_data, applying a margin to the global min and max.
    
    :param output_data: List of dictionaries, each with 'pixels' as a list of pixel values.
    :param margin_factor: Margin factor to apply to the global min and max pixel values (default 10%).
    :return: Tuple (modified output_data with normalized pixels, global min values, global max values).
    """
    all_pixels = np.array([pixel for entry in output_data for pixel in entry['pixels']])
    min_global, max_global = np.min(all_pixels, axis=0), np.max(all_pixels, axis=0)
    range_global = max_global - min_global
    min_global -= margin_factor * range_global
    max_global += margin_factor * range_global
    min_global = np.clip(min_global, 0, None)

    for entry in output_data:
        pixels_normalized = (np.array(entry['pixels']) - min_global) / (max_global - min_global)
        entry['pixels_normalized'] = pixels_normalized.tolist()

    return output_data, min_global, max_global

def realign_and_filter_pixel_data_by_bands(output_data, required_bands):
    """
    Realigns and filters pixel data in each element of output_data according to the specified required_bands.
    Raises an exception if any element lacks any of the required bands.

    :param output_data: List of dictionaries, each containing 'band_names' and 'pixels'.
    :param required_bands: List of band names to realign and filter the pixel data.
    :return: The output_data modified with pixels realigned and filtered according to required_bands.
    """
    for entry in output_data:
        band_names = entry['band_names']
        pixels = entry['pixels']

        if not all(band in band_names for band in required_bands):
            raise Exception(f"Missing required bands in the element: {band_names}")

        band_index_map = {band: band_names.index(band) for band in required_bands}
        realigned_pixels = [[pixel[band_index_map[band]] for band in required_bands] for pixel in pixels]

        entry['pixels'] = realigned_pixels
        entry['band_names'] = required_bands

    return output_data

