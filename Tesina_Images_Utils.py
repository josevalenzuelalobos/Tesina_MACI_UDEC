from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rasterio.features import geometry_mask

def robust_scaler_normalize_images(images):
    robust_scaler = RobustScaler()

    # Stack all the images into a single numpy array
    all_images = np.stack(images)

    # Retain the dimensions of the individual images for later
    image_shape = all_images.shape[1:]

    # Reshape the array so that each row is an image
    all_images = all_images.reshape(-1, np.prod(image_shape))

    # Normalize all the images at once
    all_images_normalized = robust_scaler.fit_transform(all_images)

    # Reshape the images back to their original shape
    images_normalized = [image.reshape(image_shape) for image in all_images_normalized]

    return images_normalized


def normalize_image_percentile(image, lower_percentile=1, upper_percentile=99):
    """Normaliza la imagen utilizando percentiles."""
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    image = np.clip(image, lower, upper)
    return (image - lower) / (upper - lower)

import rasterio
import rasterio.features
import geopandas as gpd

def create_mask_from_vector(polygon, raster_shape, crs, transform):
    """
    Create a raster mask from a vector polygon.

    Parameters:
    - polygon: A GeoPandas GeoSeries or similar object that contains the polygon to mask.
    - raster_shape: Tuple of the height and width of the raster.
    - crs: Coordinate reference system of the raster data.
    - transform: Affine transform for the raster data.

    Returns:
    - mask: A 2D NumPy array where pixels inside the polygon are True and all others are False.
    """
    # Ensure the polygon is in the same CRS as the raster data
    if polygon.crs != crs:
        polygon = polygon.to_crs(crs)
    
    # Rasterize the polygon to create the mask
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in polygon.geometry],
        out_shape=raster_shape,
        transform=transform,
        fill=0,  # pixels outside the polygon
        all_touched=True,  # include all pixels that touch the polygon
        dtype='uint8'
    ).astype(bool)
    
    return mask

def create_mask_from_polygon(shape, transform, width, height):
    """Create a binary mask from a polygon."""
    return geometry_mask([shape], transform=transform, invert=True, out_shape=(height, width))


def plot_distribution_before_after_robust_scaler(images, images_normalized):
    all_images = np.stack(images).reshape(-1, np.prod(np.stack(images).shape[1:]))
    all_images_normalized = np.stack(images_normalized).reshape(-1, np.prod(np.stack(images_normalized).shape[1:]))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Distribution before normalization
    sns.histplot(all_images.flatten(), ax=axs[0], color='blue', kde=True)
    axs[0].set_title('Distribution before normalization')

    # Distribution after normalization
    sns.histplot(all_images_normalized.flatten(), ax=axs[1], color='green', kde=True)
    axs[1].set_title('Distribution after normalization')

    plt.show()
    
    import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, title, x_label='Value', y_label='Frequency', bins=80, discard_outliers=0.02, discard_value: int = 0):
    """
    Plots a histogram of the given data.

    Parameters:
    data: ndarray, a 2D array where each pixel value represents a data point.
    title: str, the title of the plot.
    x_label: str, label for the x-axis.
    y_label: str, label for the y-axis.
    bins: int, the number of bins in the histogram.
    discard_outliers: float, fraction of data points to be discarded from both tails.
    """

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Flattened data
    data_flattened = data.flatten()

    # Discard the given fraction of the highest and lowest values
    data_flattened = np.sort(data_flattened)[int(len(data_flattened) * discard_outliers):int(len(data_flattened) * (1 - discard_outliers))]

    if discard_value is not None:
        data_flattened = data_flattened[data_flattened != discard_value]

    plt.figure(figsize=(15, 4))

    # Create a histogram with Seaborn
    sns.histplot(data_flattened, bins=bins, kde=False, color='skyblue', edgecolor='black')

    # Labels and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

from skimage import io
import numpy as np
import os
from typing import List

def save_images_to_folder(images: List[np.ndarray], 
                          image_names: List[str],                           
                          mask: np.ndarray = None,                           
                          folder_path: str = './', 
                          subfolder_name: str = 'images', 
                          normalize: bool=False) -> None:
    """
    Save images to folder
    :param images: images to save
    :param mask: mask to apply to images
    :param folder_path: path to folder
    :param subfolder_name: name of subfolder
    :param image_names: names of images
    :param normalize: Boolean flag to normalize image or not
    :return: None
    """
        
    path_to_folder = os.path.join(folder_path, subfolder_name)
    os.makedirs(path_to_folder, exist_ok=True)
        
    if mask is not None:
        if mask.ndim == 2:  # If the mask is a single channel
            mask = np.repeat(mask[np.newaxis, :, :], 3, axis=0)  # Duplicate the mask channel to match the image's shape


        # Save images
        for i, image in enumerate(images):
            if image.ndim == 2:  # If image is grayscale
                masked_image = image * mask[:, :, 0]  # Apply single channel mask
            elif image.ndim == 3 and image.shape[2] == 4:  # If image is RGBA
                # Apply mask to RGB channels and leave alpha channel intact
                masked_image = np.zeros_like(image)
                masked_image[:, :, :3] = image[:, :, :3] * mask
                masked_image[:, :, 3] = image[:, :, 3]
            else:  # If image is RGB
                masked_image = image * mask  # Apply multi-channel mask
                
            if normalize:
                masked_image = ((masked_image) / 
                                (masked_image.max())) * 255
            masked_image = masked_image.astype(np.uint8)
            masked_image = masked_image.transpose(1, 2, 0)
            imageio.imwrite(os.path.join(path_to_folder, image_names[i]), masked_image)
    else:
        for i, image in enumerate(images):
            if normalize:
                image = ((image) / 
                         (image.max())) * 255
            image = image.astype(np.uint8)
            image = image.transpose(1, 2, 0)
            imageio.imwrite(os.path.join(path_to_folder, image_names[i]), image)
            
    return None



def save_images_with_palette_and_labels(images, directory, filenames, palette, labels, 
                                        overwrite=False, visible_images=None, mask=None):
    
    
    # Si se proporcionan imágenes visibles, deben tener la misma longitud que las imágenes originales
    if visible_images is not None:
        assert len(images) == len(visible_images), "Images and visible_images must have the same length"
    
    if mask is not None:
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    for idx, (image, filename) in enumerate(zip(images, filenames)):
        output_path = os.path.join(directory, filename)
        
        # Check if the file exists at the beginning
        if os.path.exists(output_path) and not overwrite:
            continue
        
        rgb_image = palette[image]
         
        # Calculate aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = height / width

        # Adjust figure height based on aspect ratio
        if visible_images is not None:
            fig_width = 22
            fig_height = (fig_width * aspect_ratio) / 1.8  # We divide by 2 because two images will be shown side by side
        else:
            fig_width = 10
            fig_height = fig_width * aspect_ratio

        if mask is not None:
            assert rgb_image.shape == mask.shape, "Image and mask must have the same shape"
            rgb_image = rgb_image * mask

        plt.figure(figsize=(fig_width, fig_height))

        unique_labels, counts = np.unique(image, return_counts=True)
        total_pixels = image.size
        patches = [mpatches.Patch(color=np.array(palette[label]) / 255.,
                                  label=f"{labels[label]} [{100 * count / total_pixels:.2f}%]")
                   for label, count in zip(unique_labels, counts)]

        
        # If there's a visible_image in the list, we show it alongside the original image
        if visible_images is not None:
            visible_image = visible_images[idx].transpose(1, 2, 0)
            if mask is not None:
                visible_image = visible_image * mask
            visible_image = normalize_image_percentile(visible_image)  # Normalización usando percentiles
            plt.subplot(1, 2, 1)
            plt.imshow(visible_image)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)

        plt.imshow(rgb_image)
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.axis('off')

        plt.savefig(output_path, bbox_inches='tight')
        plt.close()