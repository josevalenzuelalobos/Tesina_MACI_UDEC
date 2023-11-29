from enum import Enum
import xarray as xr
import numpy as np


class IndexCategory(Enum):
    VEGETATION = "Vegetativos"
    FIRE = "Fuego"
    WATER = "Agua"
    CLOUDS = "Nubes"
    S2BANDS = "Bandas de Sentinel-2"
    
    @classmethod
    def get_index_category(cls, value):
        for index_category in cls:
            if isinstance(value, IndexCategory):
                comparison_value = value.value
            else:
                comparison_value = value
                
            if index_category.value == comparison_value:
                return index_category
        return None  # Return None if the value is not found in the Enum
    
    @classmethod
    def get_index_category_position(cls, value):
        for i, index_category in enumerate(cls):
            if isinstance(value, IndexCategory):
                comparison_value = value.value
            else:
                comparison_value = value
                
            if index_category.value == comparison_value:
                return i 
        return None  # Return None if the value is not found in the Enum
    
    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))
    
#Vegetation indices : DVI, RVI, PVI, IPVI, WDVI, TNDVI, GNDVI, GEMI, ARVI, NDI45, MTCI, MCARI, REIP, S2REP, IRECI, PSSRa, NDVI
class VegetationIndex(Enum):
    DVI = "DVI"   # Difference Vegetation Index: DVI = NIR - Red. Simple, may saturate in dense vegetation areas.
    RVI = "RVI"   # Ratio Vegetation Index: RVI = NIR / Red. Simple ratio, less prone to saturation.
    PVI = "PVI"   # Perpendicular Vegetation Index: PVI = c * (NIR - c * Red - b), minimizes soil effect.
    IPVI = "IPVI" # Infrared Percentage Vegetation Index: IPVI = NIR / (NIR + Red), chlorophyll content estimation.
    WDVI = "WDVI" # Weighted Difference Vegetation Index: WDVI = NIR - a * Red, soil effect reduction.
    GNDVI = "GNDVI" # Green Normalized Difference Vegetation Index: GNDVI = (NIR - Green) / (NIR + Green), NDVI variation.
    ARVI = "ARVI" # Atmospherically Resistant Vegetation Index: ARVI = (NIR - (2 * Red - Blue)) / (NIR + (2 * Red - Blue)), compensates atmospheric effects.
    MTCI = "MTCI" # MERIS Terrestrial Chlorophyll Index: MTCI = (NIR - RedEdge3) / (RedEdge3 - Red). Good for vegetation health.
    MCARI = "MCARI" # Modified Chlorophyll Absorption in Reflectance Index: MCARI = ((RedEdge - Red) - 0.2 * (RedEdge - Green)) * (RedEdge / Red), chlorophyll concentration.
    S2REP = "S2REP" # Sentinel-2 Red Edge Position: Similar to REIP but specific to Sentinel-2 data.
    IRECI = "IRECI" # Inverted Red-Edge Chlorophyll Index: Specific formula and use, may require specialized knowledge.
    PSSRa = "PSSRa" # Pigment Specific Simple Ratio: Specific to pigment changes, no standard formula.
    NDVI = "NDVI" # Normalized Difference Vegetation Index: NDVI = (NIR - Red) / (NIR + Red). Widely used, effective.
    SAVI = "SAVI" # Soil Adjusted Vegetation Index: SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L), L=0.5. Improved NDVI.
    NVR = "NVR"   # Normalized Vegetation Red Index: NVR = (NIR - Red) / (NIR + Red - Blue). NDVI variation.

    def get_position(value):
            for i, veg_index in enumerate(VegetationIndex):
                if isinstance(value, VegetationIndex):
                    comparison_value = value.value
                else:
                    comparison_value = value
                if veg_index.value == comparison_value:
                    return i 
            print(f"Value {value} not found in VegetationIndex")
            return None  # Retornar None si el valor no se encuentra en el Enum
        
    @staticmethod
    def get_index(position):
        for i, veg_index in enumerate(VegetationIndex):
            if i == position:  # Restamos 1 porque enumerate empieza desde 0
                return veg_index
        return None  # Retornar None si la posición no es válida
    
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member   #Return the member whose value is equal to the given value
        return None
        
    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))

def calculate_vegetation_index_from_xr_dataset(data: xr.Dataset, index: VegetationIndex):
    """
    Calculate vegetation index from Sentinel-2 L2A bands.
    
    Parameters:
    stac_items: xr.Dataset, Sentinel-2 L2A bands.
    index: VegetationIndex, vegetation index to calculate.

    Returns:
    index: xr.Dataset, calculated vegetation index.
    """

    if index == VegetationIndex.DVI:
         # Check that we have the necessary bands for DVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir - red
    elif index == VegetationIndex.RVI:
        # Check that we have the necessary bands for RVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir / red
    elif index == VegetationIndex.MTCI:
        # Check that we have the necessary bands for MTCI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'rededge3' in data, "Vegetation red edge 3 band is not in the dataset"
        red = data['red']
        nir = data['nir']
        rededge3 = data['rededge3']
        output = (nir - rededge3) / (rededge3 - red)
    elif index == VegetationIndex.NDVI:
        # Check that we have the necessary bands for NDVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = (nir - red) / (nir + red)
    elif index == VegetationIndex.SAVI:
        # Check that we have the necessary bands for SAVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        L = 0.5
        output = ((nir - red) / (nir + red + L)) * (1 + L)
    elif index == VegetationIndex.NVR:
        # Check that we have the necessary bands for NVR calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'blue' in data, "Blue band is not in the dataset"
        nir = data['nir']
        red = data['red']
        blue = data['blue']
        output = (nir - red) / (nir + red - blue)
    elif index == VegetationIndex.PVI:
        # Check that we have the necessary bands for PVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        c, b = 1, 0 # These constants can be adjusted depending on the application
        output = c * (nir - c * red - b)
    elif index == VegetationIndex.IPVI:
        # Check that we have the necessary bands for IPVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir / (nir + red)
    elif index == VegetationIndex.WDVI:
        # Check that we have the necessary bands for WDVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        a = 1 # This constant can be adjusted depending on the application
        output = nir - a * red
    elif index == VegetationIndex.GNDVI:
        # Check that we have the necessary bands for GNDVI calculation
        assert 'green' in data, "Green band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        green = data['green']
        nir = data['nir']
        output = (nir - green) / (nir + green)
    elif index == VegetationIndex.ARVI:
        # Check that we have the necessary bands for ARVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'blue' in data, "Blue band is not in the dataset"
        red = data['red']
        nir = data['nir']
        blue = data['blue']
        output = (nir - (2 * red - blue)) / (nir + (2 * red - blue))
    elif index == VegetationIndex.MCARI:
        # Check that we have the necessary bands for MCARI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'green' in data, "Green band is not in the dataset"
        assert 'rededge1' in data, "Red edge 1 band is not in the dataset"
        red = data['red']
        green = data['green']
        rededge1 = data['rededge1']
        output = ((rededge1 - red) - 0.2 * (rededge1 - green)) * (rededge1 / red)
    else:
        raise ValueError(f"Vegetation index {index} not implemented")
    return output

def calculate_vegetation_index_from_np_array(data: np.ndarray, index: VegetationIndex):
    """
    Calculate vegetation index from Sentinel-2 L2A bands.
    
    Parameters:
    data: np.ndarray, Sentinel-2 L2A bands.
    index: VegetationIndex, vegetation index to calculate.
    
    Returns:
    index: np.ndarray, calculated vegetation index.
    """
    
    if index == VegetationIndex.DVI:
        # Check that we have the necessary bands for DVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for DVI calculation"
        red = data[0]
        nir = data[1]
        output = nir - red
    elif index == VegetationIndex.RVI:
        # Check that we have the necessary bands for RVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for RVI calculation"
        red = data[0]
        nir = data[1]
        output = nir / red
    elif index == VegetationIndex.MTCI:
        # Check that we have the necessary bands for MTCI calculation
        assert data.shape[0] >= 3, "Red, NIR and Red edge 3 bands are required for MTCI calculation"
        red = data[0]
        nir = data[1]
        rededge3 = data[2]
        output = (nir - rededge3) / (rededge3 - red)
    elif index == VegetationIndex.NDVI:
        # Check that we have the necessary bands for NDVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for NDVI calculation"
        red = data[0]
        nir = data[1]
        output = (nir - red) / (nir + red)
    elif index == VegetationIndex.SAVI:
        # Check that we have the necessary bands for SAVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for SAVI calculation"
        red = data[0]
        nir = data[1]
        L = 0.5
        output = ((nir - red) / (nir + red + L)) * (1 + L)
    elif index == VegetationIndex.NVR:
        # Check that we have the necessary bands for NVR calculation
        assert data.shape[0] >= 3, "Red, NIR and Blue bands are required for NVR calculation"
        nir = data[0]
        red = data[1]
        blue = data[2]
        output = (nir - red) / (nir + red - blue)
    elif index == VegetationIndex.PVI:
        # Check that we have the necessary bands for PVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for PVI calculation"
        red = data[0]
        nir = data[1]
        c, b = 1, 0
        output = c * (nir - c * red - b)
    elif index == VegetationIndex.IPVI:
        # Check that we have the necessary bands for IPVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for IPVI calculation"
        red = data[0]
        nir = data[1]
        output = nir / (nir + red)
    elif index == VegetationIndex.WDVI:
        # Check that we have the necessary bands for WDVI calculation
        assert data.shape[0] >= 2, "Red and NIR bands are required for WDVI calculation"
        red = data[0]
        nir = data[1]
        a = 1
        output = nir - a * red
    elif index == VegetationIndex.GNDVI:
        # Check that we have the necessary bands for GNDVI calculation
        assert data.shape[0] >= 2, "Green and NIR bands are required for GNDVI calculation"
        green = data[0]
        nir = data[1]
        output = (nir - green) / (nir + green)
    elif index == VegetationIndex.ARVI:
        # Check that we have the necessary bands for ARVI calculation
        assert data.shape[0] >= 3, "Red, NIR and Blue bands are required for ARVI calculation"
        red = data[0]
        nir = data[1]
        blue = data[2]
        output = (nir - (2 * red - blue)) / (nir + (2 * red - blue))
    elif index == VegetationIndex.MCARI:
        # Check that we have the necessary bands for MCARI calculation
        assert data.shape[0] >= 3, "Red, Green and Red edge 1 bands are required for MCARI calculation"
        red = data[0]
        green = data[1]
        rededge1 = data[2]
        output = ((rededge1 - red) - 0.2 * (rededge1 - green)) * (rededge1 / red)
    else:
        raise ValueError(f"Vegetation index {index} not implemented")

    return output

def calculate_vegetation_index_from_dict(data: dict, index: VegetationIndex):
    """
    Calculate vegetation index from Sentinel-2 L2A bands.
    
    Parameters:
    data: dict, Sentinel-2 L2A bands.
    index: VegetationIndex, vegetation index to calculate.
    
    Returns:
    index: np.ndarray, calculated vegetation index.
    """
    
    if index == VegetationIndex.DVI:
        # Check that we have the necessary bands for DVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir - red
    elif index == VegetationIndex.RVI:
        # Check that we have the necessary bands for RVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir / red
    elif index == VegetationIndex.MTCI:
        # Check that we have the necessary bands for MTCI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'rededge3' in data, "Vegetation red edge 3 band is not in the dataset"
        red = data['red']
        nir = data['nir']
        rededge3 = data['rededge3']
        output = (nir - rededge3) / (rededge3 - red)
    elif index == VegetationIndex.NDVI:
        # Check that we have the necessary bands for NDVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = (nir - red) / (nir + red)
    elif index == VegetationIndex.SAVI:
        # Check that we have the necessary bands for SAVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        L = 0.5
        output = ((nir - red) / (nir + red + L)) * (1 + L)
    elif index == VegetationIndex.NVR:
        # Check that we have the necessary bands for NVR calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'blue' in data, "Blue band is not in the dataset"
        nir = data['nir']
        red = data['red']
        blue = data['blue']
        output = (nir - red) / (nir + red - blue)
    elif index == VegetationIndex.PVI:
        # Check that we have the necessary bands for PVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        c, b = 1, 0
        output = c * (nir - c * red - b)
    elif index == VegetationIndex.IPVI:
        # Check that we have the necessary bands for IPVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        output = nir / (nir + red)
    elif index == VegetationIndex.WDVI:
        # Check that we have the necessary bands for WDVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        red = data['red']
        nir = data['nir']
        a = 1
        output = nir - a * red
    elif index == VegetationIndex.GNDVI:
        # Check that we have the necessary bands for GNDVI calculation
        assert 'green' in data, "Green band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        green = data['green']
        nir = data['nir']
        output = (nir - green) / (nir + green)
    elif index == VegetationIndex.ARVI:
        # Check that we have the necessary bands for ARVI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'nir' in data, "NIR band is not in the dataset"
        assert 'blue' in data, "Blue band is not in the dataset"
        red = data['red']
        nir = data['nir']
        blue = data['blue']
        output = (nir - (2 * red - blue)) / (nir + (2 * red - blue))
    elif index == VegetationIndex.MCARI:
        # Check that we have the necessary bands for MCARI calculation
        assert 'red' in data, "Red band is not in the dataset"
        assert 'green' in data, "Green band is not in the dataset"
        assert 'rededge1' in data, "Red edge 1 band is not in the dataset"
        red = data['red']
        green = data['green']
        rededge1 = data['rededge1']
        output = ((rededge1 - red) - 0.2 * (rededge1 - green)) * (rededge1 / red)
    else:
        raise ValueError(f"Vegetation index {index} not implemented")

    return output
    

class WaterIndex(Enum):
    NDWI = "NDWI"   # Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR). Useful for open water detection.
    MNDWI = "MNDWI" # Modified NDWI: MNDWI = (Green - SWIR) / (Green + SWIR). Enhanced water features, especially turbid water.
    AWEI_ns = "AWEI_ns" # Automated Water Extraction Index, no shadows: AWEI_ns = 4 * (Green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2). Water bodies without shadow effect.
    AWEI_sh = "AWEI_sh" # Automated Water Extraction Index, shadows: AWEI_sh = Blue + 2.5 * Green - 1.5 * (NIR + SWIR1) - 0.25 * SWIR2. Differentiate shadow and water.
    WI = "WI"       # Water Index: WI = Green + 2.5 * Blue - 1.5 * (NIR + Red) - 0.25 * SWIR. General water detection.
    LSWI = "LSWI"   # Land Surface Water Index: LSWI = (NIR - SWIR) / (NIR + SWIR). Highlights water content in land surface.
    WRI = "WRI"     # Water Ratio Index: WRI = (Green + Red) / NIR. Designed for inland water bodies.
    NDSI = "NDSI"   # Normalized Difference Snow Index: NDSI = (Green - SWIR) / (Green + SWIR). Useful for snow but can be adapted for water.
    
    @staticmethod
    def get_position(value):
            for i, water_index in enumerate(WaterIndex):
                if isinstance(value, WaterIndex):
                    comparison_value = value.value
                else:
                    comparison_value = value
                    
                if water_index.value == comparison_value:
                    return i 
            return None  # Retornar None si el valor no se encuentra en el Enum
        
    @staticmethod
    def get_index(position):
        for i, water_index in enumerate(WaterIndex):
            if i == position:  # Restamos 1 porque enumerate empieza desde 0
                return water_index
        return None  # Retornar None si la posición no es válida
    
    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member   #Return the member whose value is equal to the given value
        return None
    
def calculate_water_index_from_xr_dataset(data: xr.Dataset, index: WaterIndex):
    """
    Calculate water index from Sentinel-2 L2A bands.

    Parameters:
    data: xr.Dataset, Sentinel-2 L2A bands.
    index: WaterIndex, water index to calculate.

    Returns:
    output: xr.Dataset, calculated water index.
    """

    if index == WaterIndex.NDWI:
        # Check if the bands are in the dataset
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."        
        green = data['green'].astype('int16')
        nir = data['nir'].astype('int16')
        output = (green - nir) / (green + nir)
    elif index == WaterIndex.MNDWI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        output = (green - swir16) / (green + swir16)
    elif index == WaterIndex.AWEI_ns:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        green = data['green'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        nir = data['nir'].astype('int16')
        output = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir16)
    elif index == WaterIndex.AWEI_sh:
        assert 'blue' in data, "Band 'blue' is not in the dataset."
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        blue = data['blue'].astype('int16')
        green = data['green'].astype('int16')
        nir = data['nir'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        swir22 = data['swir22'].astype('int16')
        output = blue + 2.5 * green - 1.5 * (nir + swir16) - 0.25 * swir22
    elif index == WaterIndex.WI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'blue' in data, "Band 'blue' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green'].astype('int16')
        blue = data['blue'].astype('int16')
        nir = data['nir'].astype('int16')
        red = data['red'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        output = green + 2.5 * blue - 1.5 * (nir + red) - 0.25 * swir16
    elif index == WaterIndex.LSWI:
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        nir = data['nir'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        output = (nir - swir16) / (nir + swir16)
    elif index == WaterIndex.WRI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        green = data['green'].astype('int16')
        red = data['red'].astype('int16')
        nir = data['nir'].astype('int16')
        output = (green + red) / nir
    elif index == WaterIndex.NDSI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        output = (green - swir16) / (green + swir16)
    else:
        raise ValueError(f"Water index {index} not implemented")

    return output

def calculate_water_index_from_np_array(data: np.ndarray, index: WaterIndex):
    """
    Calculate water index from Sentinel-2 L2A bands.

    Parameters:
    data: np.ndarray, Sentinel-2 L2A bands.
    index: WaterIndex, water index to calculate.
    
    Returns:
    index: np.ndarray, calculated water index.
    """
    
    if index == WaterIndex.NDWI:
        # Check if the bands are in the dataset
        assert data.shape[0] >= 2, "Green and NIR bands are required for NDWI calculation"
        green = data[0]
        nir = data[1]
        output = (green - nir) / (green + nir)
    elif index == WaterIndex.MNDWI:
        assert data.shape[0] >= 2, "Green and SWIR bands are required for MNDWI calculation"
        green = data[0]
        swir16 = data[1]
        output = (green - swir16) / (green + swir16)
    elif index == WaterIndex.AWEI_ns:
        assert data.shape[0] >= 3, "Green, SWIR1 and NIR bands are required for AWEI_ns calculation"
        green = data[0]
        swir16 = data[1]
        nir = data[2]
        output = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir16)
    elif index == WaterIndex.AWEI_sh:
        assert data.shape[0] >= 5, "Blue, Green, NIR, SWIR1 and SWIR2 bands are required for AWEI_sh calculation"
        blue = data[0]
        green = data[1]
        nir = data[2]
        swir16 = data[3]
        swir22 = data[4]
        output = blue + 2.5 * green - 1.5 * (nir + swir16) - 0.25 * swir22
    elif index == WaterIndex.WI:
        assert data.shape[0] >= 5, "Green, Blue, NIR, Red and SWIR bands are required for WI calculation"
        green = data[0]
        blue = data[1]
        nir = data[2]
        red = data[3]
        swir16 = data[4]
        output = green + 2.5 * blue - 1.5 * (nir + red) - 0.25 * swir16
    elif index == WaterIndex.LSWI:
        assert data.shape[0] >= 2, "NIR and SWIR bands are required for LSWI calculation"
        nir = data[0]
        swir16 = data[1]
        output = (nir - swir16) / (nir + swir16)
    elif index == WaterIndex.WRI:
        assert data.shape[0] >= 3, "Green, Red and NIR bands are required for WRI calculation"
        green = data[0]
        red = data[1]
        nir = data[2]
        output = (green + red) / nir
    elif index == WaterIndex.NDSI:
        assert data.shape[0] >= 2, "Green and SWIR bands are required for NDSI calculation"
        green = data[0]
        swir16 = data[1]
        output = (green - swir16) / (green + swir16)
    else:
        raise ValueError(f"Water index {index} not implemented")

    return output

def calculate_water_index_from_dict(data: dict, index: WaterIndex):
    """
    Calculate water index from Sentinel-2 L2A bands.

    Parameters:
    data: dict, Sentinel-2 L2A bands.
    index: WaterIndex, water index to calculate.
    
    Returns:
    index: np.ndarray, calculated water index.
    """
    
    if index == WaterIndex.NDWI:
        # Check if the bands are in the dataset
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."        
        green = data['green']
        nir = data['nir']
        output = (green - nir) / (green + nir)
    elif index == WaterIndex.MNDWI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green']
        swir16 = data['swir16']
        output = (green - swir16) / (green + swir16)
    elif index == WaterIndex.AWEI_ns:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        green = data['green']
        swir16 = data['swir16']
        nir = data['nir']
        output = 4 * (green - swir16) - (0.25 * nir + 2.75 * swir16)
    elif index == WaterIndex.AWEI_sh:
        assert 'blue' in data, "Band 'blue' is not in the dataset."
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        blue = data['blue']
        green = data['green']
        nir = data['nir']
        swir16 = data['swir16']
        swir22 = data['swir22']
        output = blue + 2.5 * green - 1.5 * (nir + swir16) - 0.25
    elif index == WaterIndex.WI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'blue' in data, "Band 'blue' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green']
        blue = data['blue']
        nir = data['nir']
        red = data['red']
        swir16 = data['swir16']
        output = green + 2.5 * blue - 1.5 * (nir + red) - 0.25 * swir16
    elif index == WaterIndex.LSWI:
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        nir = data['nir']
        swir16 = data['swir16']
        output = (nir - swir16) / (nir + swir16)
    elif index == WaterIndex.WRI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        green = data['green']
        red = data['red']
        nir = data['nir']
        output = (green + red) / nir
    elif index == WaterIndex.NDSI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        green = data['green']
        swir16 = data['swir16']
        output = (green - swir16) / (green + swir16)
    else:
        raise ValueError(f"Water index {index} not implemented")
    
    return output

    

class FireIndex(Enum):
    NBR = "NBR"    # Normalized Burn Ratio: (NIR - SWIR) / (NIR + SWIR). Used to highlight burned areas.
    dNBR = "dNBR"  # Difference Normalized Burn Ratio: (NBR_pre - NBR_post). Highlights changes due to fire.
    SAVI = "SAVI"  # Soil Adjusted Vegetation Index: ((NIR - Red) / (NIR + Red + L)) * (1 + L). Useful for burned vegetation assessment.
    MIRBI = "MIRBI" # Mid-Infrared Burn Index: (10 * SWIR) - (9.8 * NIR) + 2. Useful for burned area mapping.
    BAI = "BAI"    # Burn Area Index: (1.0 / ((0.10 - Red) ** 2 + (0.06 - NIR) ** 2)). Used to highlight burned areas.
    AFI = "AFI"    # Active Fire Index: (SWIR2 - SWIR1) / (SWIR2 + SWIR1). Used to highlight active fires.
    VFDI = "VFDI"  # Vegetative Fire Detection Index: (SWIR2 + NIR) / (SWIR2 - NIR). Used to highlight active fires.
    MSRIF = "MSRIF" # Modified SWIR Ratio Index for Fire: (SWIR2/SWIR1) - 1. Used to highlight burned areas.
    SNDI = "SNDI"  # SWIR-NIR Detection Index: (SWIR1 + SWIR2) / (NIR). Used to highlight active fires.

    @staticmethod
    def get_position(value):
            for i, fire_index in enumerate(FireIndex):
                if isinstance(value, FireIndex):
                    comparison_value = value.value
                else:
                    comparison_value = value
                    
                if fire_index.value == comparison_value:
                    return i 
            return None  # Retornar None si el valor no se encuentra en el Enum
        
    @staticmethod
    def get_index(position):
        for i, fire_index in enumerate(FireIndex):
            if i == position:  # Restamos 1 porque enumerate empieza desde 0
                return fire_index
        return None  # Retornar None si la posición no es válida
    
    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member   #Return the member whose value is equal to the given value
        return None
    
     
def calculate_fire_index_from_xr_dataset(data: xr.Dataset, index: FireIndex):
    """
    Calculate fire index from Sentinel-2 L2A bands.

    Parameters:
    data: xr.Dataset, Sentinel-2 L2A bands.
    index: FireIndex, fire index to calculate.

    Returns:
    output: xr.Dataset, calculated fire index.
    """

    if index == FireIndex.NBR:
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        nir = data['nir'].astype('int16')
        swir22 = data['swir22'].astype('int16')
        output = (nir - swir22) / (nir + swir22)
    elif index == FireIndex.dNBR:
        # You will need to provide both pre-fire and post-fire NBR datasets
        assert 'nbr_pre' in data, "Band 'nbr_pre' is not in the dataset."
        assert 'nbr_post' in data, "Band 'nbr_post' is not in the dataset."
        nbr_pre = data['nbr_pre'].astype('int16')
        nbr_post = data['nbr_post'].astype('int16')
        output = nbr_pre - nbr_post
    elif index == FireIndex.SAVI:
        L = 0.5  # Soil brightness correction factor
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        nir = data['nir'].astype('int16')
        red = data['red'].astype('int16')
        output = ((nir - red) / (nir + red + L)) * (1 + L)
    elif index == FireIndex.MIRBI:
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        swir22 = data['swir22'].astype('int16')
        nir = data['nir'].astype('int16')
        output = (10 * swir22) - (9.8 * nir) + 2
    elif index == FireIndex.BAI:
        assert 'red' in data, "Band 'red' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        red = data['red'].astype('int16')
        nir = data['nir'].astype('int16')
        output = 1.0 / ((0.10 - red) ** 2 + (0.06 - nir) ** 2)
    elif index == FireIndex.AFI:
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        swir22 = data['swir22'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        output = (swir22 - swir16) / (swir22 + swir16)
    elif index == FireIndex.VFDI:
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        swir16 = data['swir16'].astype('int16')
        nir = data['nir'].astype('int16')
        output = (swir16 + nir) / (swir16 - nir)
    elif index == FireIndex.MSRIF:
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        swir16 = data['swir16'].astype('int16')
        swir22 = data['swir22'].astype('int16')
        output = (swir22 / swir16) - 1
    elif index == FireIndex.SNDI:
        assert 'swir22' in data, "Band 'swir22' is not in the dataset."
        assert 'swir16' in data, "Band 'swir16' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        swir22 = data['swir22'].astype('int16')
        swir16 = data['swir16'].astype('int16')
        nir = data['nir'].astype('int16')
        output = (swir16 + swir22) / nir
    else:
        raise ValueError(f"Fire index {index} not implemented")
    return output


def calculate_fire_index_from_np_array(data: np.ndarray, index: FireIndex):
    """
    Calculate fire index from Sentinel-2 L2A bands.

    Parameters:
    data: np.ndarray, Sentinel-2 L2A bands.
    index: FireIndex, fire index to calculate.
    
    Returns:
    output: np.ndarray, calculated fire index.
    """
    
    if index == FireIndex.NBR:
        assert data.shape[0] == 2, "NBR requires 2 bands"
        output = (data[0] - data[1]) / (data[0] + data[1])
    elif index == FireIndex.dNBR:
        assert data.shape[0] == 2, "dNBR requires 2 bands"
        output = data[0] - data[1]
    elif index == FireIndex.SAVI:
        assert data.shape[0] == 2, "SAVI requires 2 bands"
        L = 0.5  # Soil brightness correction factor
        output = ((data[0] - data[1]) / (data[0] + data[1] + L)) * (1 + L)
    elif index == FireIndex.MIRBI:
        assert data.shape[0] == 2, "MIRBI requires 2 bands"
        output = (10 * data[0]) - (9.8 * data[1]) + 2
    elif index == FireIndex.BAI:
        assert data.shape[0] == 2, "BAI requires 2 bands"
        output = 1.0 / ((0.10 - data[0]) ** 2 + (0.06 - data[1]) ** 2)
    elif index == FireIndex.AFI:
        assert data.shape[0] == 2, "AFI requires 2 bands"
        output = (data[0] - data[1]) / (data[0] + data[1])
    elif index == FireIndex.VFDI:
        assert data.shape[0] == 2, "VFDI requires 2 bands"
        output = (data[0] + data[1]) / (data[0] - data[1])
    elif index == FireIndex.MSRIF:
        assert data.shape[0] == 2, "MSRIF requires 2 bands"
        output = (data[0] / data[1]) - 1
    elif index == FireIndex.SNDI:
        assert data.shape[0] == 3, "SNDI requires 3 bands"
        output = (data[0] + data[1]) / data[2]
    else:
        raise ValueError(f"Fire index {index } not implemented")
    
    return output
    
def calculate_fire_index_from_dict(data: dict, index: FireIndex):
    """
    Calculate fire index from Sentinel-2 L2A bands.

    Parameters:
    data: dict, Sentinel-2 L2A bands.
    index: FireIndex, fire index to calculate.
    
    Returns:
    output: np.ndarray, calculated fire index.
    """
        
    if index == FireIndex.NBR:
        assert len(data) == 2, "NBR requires 2 bands"
        output = (data['nir'] - data['swir22']) / (data['nir'] + data['swir22'])
    elif index == FireIndex.dNBR:
        assert len(data) == 2, "dNBR requires 2 bands"
        output = data['nbr_pre'] - data['nbr_post']
    elif index == FireIndex.SAVI:
        assert len(data) == 2, "SAVI requires 2 bands"
        L = 0.5  # Soil brightness correction factor
        output = ((data['nir'] - data['red']) / (data['nir'] + data['red'] + L)) * (1 + L)
    elif index == FireIndex.MIRBI:
        assert len(data) == 2, "MIRBI requires 2 bands"
        output = (10 * data['swir22']) - (9.8 * data['nir']) + 2
    elif index == FireIndex.BAI:
        assert len(data) == 2, "BAI requires 2 bands"
        output = 1.0 / ((0.10 - data['red']) ** 2 + (0.06 - data['nir']) ** 2)
    elif index == FireIndex.AFI:
        assert len(data) == 2, "AFI requires 2 bands"
        output = (data['swir22'] - data['swir16']) / (data['swir22'] + data['swir16'])
    elif index == FireIndex.VFDI:
        assert len(data) == 2, "VFDI requires 2 bands"
        output = (data['swir16'] + data['nir']) / (data['swir16'] - data['nir'])
    elif index == FireIndex.MSRIF:
        assert len(data) == 2, "MSRIF requires 2 bands"
        output = (data['swir22'] / data['swir16']) - 1
    elif index == FireIndex.SNDI:
        assert len(data) == 3, "SNDI requires 3 bands"
        output = (data['swir16'] + data['swir22']) / data['nir']
    else:
        raise ValueError(f"Fire index {index } not implemented")
    
    return output

class CloudIndex(Enum):
    CCI = "CCI"   # Cloud Cover Index: CCI = (SWIR - NIR) / (SWIR + NIR). Useful for cloud detection and classification.
    NDCI = "NDCI" # Normalized Difference Cloud Index: NDCI = (Green - Red) / (Green + Red). Useful for cloud detection.
    
    @staticmethod
    def get_position(value):
        for i, cloud_index in enumerate(CloudIndex):
            if isinstance(value, CloudIndex):
                comparison_value = value.value
            else:
                comparison_value = value
                
            if cloud_index.value == comparison_value:
                return i 
        return None  # Return None if the value is not found in the Enum
    
    @staticmethod
    def get_index(position):
        for i, cloud_index in enumerate(CloudIndex):
            if i == position: 
                return cloud_index
        return None  # Return None if the position is not valid
    
    @classmethod
    def to_list(cls):
        return list(map(lambda c: c.value, cls))
    
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member   #Return the member whose value is equal to the given value
        return None
    
def calculate_cloud_index_from_xr_dataset(data: xr.Dataset, index: CloudIndex):
    """
    Calculate cloud index from satellite bands.

    Parameters:
    data: xr.Dataset, satellite bands.
    index: CloudIndex, cloud index to calculate.

    Returns:
    output: xr.Dataset, calculated cloud index.
    """

    if index == CloudIndex.CCI:
        assert 'swir16' in data, "Band 'swir' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        swir = data['swir16']
        nir = data['nir']
        output = (swir - nir) / (swir + nir)
    elif index == CloudIndex.NDCI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        green = data['green'].astype('float32')
        red = data['red'].astype('float32')
        output = (green - red) / (green + red)

    return output

def calculate_cloud_index_from_np_array(data: np.ndarray, index: CloudIndex):
    """
    Calculate cloud index from satellite bands.

    Parameters:
    data: np.ndarray, satellite bands.
    index: CloudIndex, cloud index to calculate.
    
    Returns:
    output: np.ndarray, calculated cloud index.
    """
    
    if index == CloudIndex.CCI:
        assert data.shape[0] >= 2, "SWIR and NIR bands are required for CCI calculation"
        swir = data[0]
        nir = data[1]
        output = (swir - nir) / (swir + nir)
    elif index == CloudIndex.NDCI:
        assert data.shape[0] >= 2, "Green and Red bands are required for NDCI calculation"
        green = data[0]
        red = data[1]
        output = (green - red) / (green + red)

    return output

def calculate_cloud_index_from_dict(data: dict, index: CloudIndex):
    """
    Calculate cloud index from satellite bands.

    Parameters:
    data: dict, satellite bands.
    index: CloudIndex, cloud index to calculate.
    
    Returns:
    output: np.ndarray, calculated cloud index.
    """
    
    if index == CloudIndex.CCI:
        assert 'swir16' in data, "Band 'swir' is not in the dataset."
        assert 'nir' in data, "Band 'nir' is not in the dataset."
        swir = data['swir16']
        nir = data['nir']
        output = (swir - nir) / (swir + nir)
    elif index == CloudIndex.NDCI:
        assert 'green' in data, "Band 'green' is not in the dataset."
        assert 'red' in data, "Band 'red' is not in the dataset."
        green = data['green']
        red = data['red']
        output = (green - red) / (green + red)

    return output
    
def calculate_vegetation_index(data, index: VegetationIndex):
    if isinstance(data, xr.Dataset):
        return calculate_vegetation_index_from_xr_dataset(data, index)
    elif isinstance(data, np.ndarray):
        return calculate_vegetation_index_from_np_array(data, index)
    elif isinstance(data, dict):
        return calculate_vegetation_index_from_dict(data, index)
    else:
        raise TypeError("Unsupported data type for vegetation index calculation")
    
def calculate_water_index(data, index: WaterIndex):
    if isinstance(data, xr.Dataset):
        return calculate_water_index_from_xr_dataset(data, index)
    elif isinstance(data, np.ndarray):
        return calculate_water_index_from_np_array(data, index)
    elif isinstance(data, dict):
        return calculate_water_index_from_dict(data, index)
    else:
        raise TypeError("Unsupported data type for water index calculation")

def calculate_fire_index(data, index: FireIndex):
    if isinstance(data, xr.Dataset):
        return calculate_fire_index_from_xr_dataset(data, index)
    elif isinstance(data, np.ndarray):
        return calculate_fire_index_from_np_array(data, index)
    elif isinstance(data, dict):
        return calculate_fire_index_from_dict(data, index)
    else:
        raise TypeError("Unsupported data type for fire index calculation")

def calculate_cloud_index(data, index: CloudIndex):
    if isinstance(data, xr.Dataset):
        return calculate_cloud_index_from_xr_dataset(data, index)
    elif isinstance(data, np.ndarray):
        return calculate_cloud_index_from_np_array(data, index)
    elif isinstance(data, dict):
        return calculate_cloud_index_from_dict(data, index)
    else:
        raise TypeError("Unsupported data type for cloud index calculation")
    
    
class ForestCondition(Enum):
    DEF = "DEF"  # Deforestation: Areas where trees have been removed or cut down.
    BUR = "BUR"  # Burned: Areas affected by fire, either naturally occurring or man-made.
    PLAG = "PLAG" # Plague: Areas affected by pests or diseases that harm the vegetation.
    OTH = "OTH"  # Other: Any other condition not covered by the previous categories.
    
    
def check_indices_availability(selected_bands, vegetation=True, water=True, fire=True, cloud=True):
    """
    Check which indices are calculable with the given set of bands by trying to calculate each index
    and catching exceptions.

    Parameters:
    - selected_bands: set of str, names of the available bands.
    - vegetation: bool, whether to check vegetation indices.
    - water: bool, whether to check water indices.
    - fire: bool, whether to check fire indices.
    - cloud: bool, whether to check cloud indices.

    Returns:
    - available_indices: list, indices that can be calculated with the given bands.
    - unavailable_indices: list, indices that cannot be calculated with the given bands.
    """

    available_indices = []
    unavailable_indices = []

    # Create dummy data based on the selected bands, using zero arrays for simplicity
    dummy_data = {band: np.array([0]) for band in selected_bands}

    def check_index(index_enum, calculate_func):
        for index in index_enum:
            try:
                # Attempt to calculate the index with dummy data
                calculate_func(dummy_data, index)
                available_indices.append(index)
            except (AssertionError, KeyError):
                # If an exception is raised, the index cannot be calculated with the selected bands
                unavailable_indices.append(index)

    # Check each category of indices if enabled
    if vegetation:
        check_index(VegetationIndex, calculate_vegetation_index_from_dict)

    if water:
        check_index(WaterIndex, calculate_water_index_from_dict)

    if fire:
        check_index(FireIndex, calculate_fire_index_from_dict)

    if cloud:
        check_index(CloudIndex, calculate_cloud_index_from_dict)

    return available_indices, unavailable_indices

