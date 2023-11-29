from typing import Any, Optional, Tuple
import cv2
import numpy as np

def polygon_coords_to_px_coords(polygon, width: int, height: int) -> list[Tuple[int, int]]:
    """Transforms polygon coordinates from lat/lon to pixel coordinates

    :param polygon: list of coordinates in lat/lon
    :param bbox: bounding box
    :param width: width of bounding box in pixels
    :param height: height of bounding box in pixels
    :return: list of coordinates in pixels
    """
    east1, north1 = polygon.bounds[0], polygon.bounds[1]
    east2, north2 = polygon.bounds[2], polygon.bounds[3]
    div_x = (east2 - east1) / width
    div_y = (north2 - north1) / height

    pixel_coords = []

    for coord in polygon.exterior.coords:
        # Scale and translate the coordinate
        px_coord = ((coord[0] - east1) / div_x, (coord[1] - north1) / div_y)
        # Append to the list of pixel coordinates
        pixel_coords.append(px_coord)
    return pixel_coords

def get_kml_polygon_masks(polygon, width: int, height: int) -> list[Tuple[np.ndarray, np.ndarray]]:
    #get the polygon mask
    polygon_coords = polygon_coords_to_px_coords(polygon, width, height)
    polygon_mask = np.zeros((height, width))
    polygon_mask = cv2.fillPoly(polygon_mask, np.array([polygon_coords], dtype=np.int32), 1)
    polygon_mask = np.flipud(polygon_mask).astype(np.uint8)
    return polygon_mask, np.dstack([polygon_mask] * 3)