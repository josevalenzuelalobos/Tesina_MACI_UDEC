from enum import Enum
class S2Band(Enum):
    B01 = "coastal"    # Aerosols, 443 nm
    B02 = "blue"       # Blue, 492.4 nm (S2A), 492.1 nm (S2B)
    B03 = "green"      # Green, 559.8 nm (S2A), 559 nm (S2B)
    B04 = "red"        # Red, 664.6 nm (S2A), 665 nm (S2B)
    B05 = "rededge1"   # Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)
    B06 = "rededge2"   # Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)
    B07 = "rededge3"   # Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)
    B08 = "nir"        # NIR, 832.8 nm (S2A), 833 nm (S2B)
    B8A = "nir08"      # Narrow NIR, 864.7 nm (S2A), 864 nm (S2B)
    B09 = "nir09"      # Water vapour, 945 nm (S2A), 943.2 nm (S2B)
    # Asumiendo que swir16 y swir22 se refieren a otras bandas SWIR:
    B11 = "swir16"     # SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)
    B12 = "swir22"     # SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)
    
    @classmethod
    def get_band(cls, band_name: str):
        for band in cls:
            if band.value == band_name:
                return band
        raise ValueError(f"Band {band_name} not found.")
    
    @classmethod
    def get_band_position(cls, value):
        if isinstance(value, S2Band):
            comparison_value = value.value
        elif isinstance(value, str):
            comparison_value = value
        else:
            raise ValueError(f"Value must be of type S2Band or str, not {type(value)}")

        for i, band in enumerate(cls):
            if band.value == comparison_value:
                return i
        raise ValueError(f"Band {value} not found.")
    
    @classmethod
    def from_position(cls, position):
        for i, band in enumerate(cls):
            if i == position:
                return band
        raise ValueError(f"Band at position {position} not found.")
    
    @classmethod
    def to_list(cls):
        return [band.value for band in cls]
        
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member   #Return the member whose value is equal to the given value
        return None
    
class S2BandNames(Enum):
    AEROSOLS = 0,
    BLUE = 1,
    GREEN = 2,
    RED = 3,
    VEGETATION_RED_EDGE_1 = 4,
    VEGETATION_RED_EDGE_2 = 5,
    VEGETATION_RED_EDGE_3 = 6,
    NIR = 7,
    NARROW_NIR = 8,
    WATER_VAPOUR = 9,
    SWIR = 10,
    SWIR_2 = 11

    
import xarray as xr
import numpy as np
from typing import List, Tuple, Optional
from Tesina_General_Utils import Logger, LogLevel

def classify_sunny_cloudy_dates_by_scene_classification(ds: xr.Dataset, thresholds: dict, mask: Optional[np.ndarray] = None) -> Tuple[List[str], List[str]]:
    """
    Classify dates in xarray dataset as sunny or cloudy based on the SCL band and thresholds.

    Args:
    - ds (xr.Dataset): xarray dataset containing Sentinel-2 bands.
    - thresholds (dict): Dictionary containing cloud coverage criteria.
    - mask (np.ndarray, optional): Binary mask to filter the region of interest. Default is None.

    Returns:
    - Tuple[List[str], List[str]]: Two lists of dates categorized as sunny and cloudy.
    """
    
    logger = Logger(min_log_level=LogLevel.Info)
    sunny_dates = []
    cloudy_dates = []

    for t in ds.time.values:
        # Calculate percentages for the specific date
        current_scl = ds['scl'].sel(time=t).data
        
        # Apply the mask to the current_scl if provided
        if mask is not None:
            masked_scl = current_scl * mask
        else:
            masked_scl = current_scl
        
        if mask is not None:
            total_pixels = np.sum(mask)  # Total pixels are the sum of the mask
        else:
            total_pixels = current_scl.size  # Original total pixels

        cloud_cover_percent = 100 * (np.sum((masked_scl == 7) | (masked_scl == 8) | (masked_scl == 9)) / total_pixels)
        low_proba_clouds_percent = 100 * (np.sum(masked_scl == 7) / total_pixels)
        medium_proba_clouds_percent = 100 * (np.sum(masked_scl == 8) / total_pixels)
        high_proba_clouds_percent = 100 * (np.sum(masked_scl == 9) / total_pixels)
        thin_cirrus_percent = 100 * (np.sum(masked_scl == 10) / total_pixels)
        cloud_shadow_percent = 100 * (np.sum(masked_scl == 3) / total_pixels)

        if any([cloud_cover_percent, medium_proba_clouds_percent, high_proba_clouds_percent, thin_cirrus_percent, cloud_shadow_percent]):
            logger.debug(f"Date: {t}. Cloud cover: {cloud_cover_percent}. Medium proba clouds: {medium_proba_clouds_percent}. High proba clouds: {high_proba_clouds_percent}. Thin cirrus: {thin_cirrus_percent}. Cloud shadow: {cloud_shadow_percent}")
        
        properties = {
            'eo:cloud_cover': cloud_cover_percent,
            's2:low_proba_clouds_percentage': low_proba_clouds_percent, 
            's2:medium_proba_clouds_percentage': medium_proba_clouds_percent,
            's2:high_proba_clouds_percentage': high_proba_clouds_percent,
            's2:thin_cirrus_percentage': thin_cirrus_percent,
            's2:cloud_shadow_percentage': cloud_shadow_percent
        }

        # Check thresholds
        if all(properties[key] < value for key, value in thresholds.items()):
            sunny_dates.append(str(t))
        else:
            cloudy_dates.append(str(t))
            logger.trace(f"Cloudy date: {t}")
            for key, value in thresholds.items():
                if properties[key] >= value:
                    logger.debug(f"Cloudy date: {t}. Coverage {key} exceeded threshold {value}, value was {properties[key]}")
                    
    # Convert to datetime64
    sunny_dates = np.array(sunny_dates, dtype='datetime64')
    cloudy_dates = np.array(cloudy_dates, dtype='datetime64')

    return sunny_dates, cloudy_dates