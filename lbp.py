from skimage.feature import local_binary_pattern
import numpy as np
from pathlib import Path

from typing import Dict


def compute_lbp_for_single_image(
    image_name: str, image_value: np.ndarray, radius: int, n_points: int, method: str = "nri_uniform"
) -> np.ndarray:
    """
    Compute LBP histogram for a single image with caching support.
    
    Args:
        image_name (str): Name of the image file (without extension)
        image_value (np.ndarray): Image array
        radius (int): Radius for LBP computation
        n_points (int): Number of points for LBP computation
        method (str): LBP method (default: "nri_uniform")
        
    Returns:
        np.ndarray: LBP histogram
    """
    # Define cache directory and file path
    cache_dir = Path("features/lbps")
    cache_file = cache_dir / f"{image_name}_lbp.npy"
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            histo = np.load(cache_file)
            return histo
        except Exception as e:
            print(f"Warning: Failed to load cached LBP for {image_name}: {e}")
            # Continue to compute LBP if loading fails
    
    # Compute LBP if not cached or loading failed
    lbp = local_binary_pattern(image_value, n_points, radius, method)
    histo = build_histogram_from_lbp(lbp, n_points)
    
    # Save to cache
    try:
        np.save(cache_file, histo)
    except Exception as e:
        print(f"Warning: Failed to save LBP cache for {image_name}: {e}")
    
    return histo


def compute_lbp_for_many_images(
    images: Dict[str, np.ndarray],
    radius: int,
    n_points: int,
    method: str = "nri_uniform",
) -> Dict[str, np.ndarray]:
    lbps = {}

    for name, img in images.items():
        lbps[name] = compute_lbp_for_single_image(name, img, radius, n_points, method)

    return lbps


def compute_lbp_for_each_category(
    categories: list[str],
    images: Dict[str, Dict[str, np.ndarray]],
    radius=2,
    n_points=8,
    method="nri_uniform",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute LBP for each category of regular images.
    
    Args:
        categories: List of category names
        images: Dict structure {category: {image_name: image_array}}
        radius: LBP radius parameter
        n_points: LBP n_points parameter  
        method: LBP method
        
    Returns:
        Dict structure {category: {image_name: lbp_histogram}}
    """
    return {
        category: compute_lbp_for_many_images(
            images[category], radius, n_points, method
        )
        for category in categories
    }


def compute_lbp_for_segmented_image(
    image_name: str, 
    regions_matrix: np.ndarray, 
    radius: int, 
    n_points: int, 
    method: str = "nri_uniform"
) -> np.ndarray:
    """
    Compute LBP features for all segments of a single image with caching support.
    
    Args:
        image_name (str): Name of the image file (without extension)
        regions_matrix (np.ndarray): Matrix containing image segments
        radius (int): Radius for LBP computation
        n_points (int): Number of points for LBP computation
        method (str): LBP method (default: "nri_uniform")
        
    Returns:
        np.ndarray: Array containing LBP histograms for all segments
    """
    # Define cache directory and file path
    cache_dir = Path("features/lbps")
    cache_file = cache_dir / f"{image_name}_segmented_lbp.npy"
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            features_array = np.load(cache_file)
            return features_array
        except Exception as e:
            print(f"Warning: Failed to load cached segmented LBP for {image_name}: {e}")
            # Continue to compute LBP if loading fails
    
    # Compute LBP for all segments
    features_list = []
    rows, cols = regions_matrix.shape
    
    for row in range(rows):
        for col in range(cols):
            region = regions_matrix[row, col]
            if _is_valid_region(region):
                try:
                    lbp = local_binary_pattern(region, n_points, radius, method)
                    histo = build_histogram_from_lbp(lbp, n_points)

                    features_list.append(histo)
                except Exception as e:
                    print(f"Warning: Failed to compute LBP for {image_name} region ({row},{col}): {e}")
                    # Add zero array as placeholder for failed regions
                    n_bins = 2**n_points if method == "uniform" else n_points + 2
                    features_list.append(np.zeros(n_bins))
    
    # Convert to numpy array
    if features_list:
        # Check if all histograms have the same shape
        if len(features_list) > 1:
            shapes = [len(hist) for hist in features_list]
            max_shape = max(shapes)
            
            # Pad histograms to have the same size
            normalized_features = []
            for hist in features_list:
                if len(hist) < max_shape:
                    # Pad with zeros
                    padded_hist = np.zeros(max_shape)
                    padded_hist[:len(hist)] = hist
                    normalized_features.append(padded_hist)
                else:
                    normalized_features.append(hist)
            
            features_array = np.array(normalized_features)
        else:
            features_array = np.array(features_list)
    else:
        # Fallback for completely empty image
        n_bins = 2**n_points if method == "uniform" else n_points + 2
        features_array = np.array([np.zeros(n_bins)])
    
    # Save to cache
    try:
        np.save(cache_file, features_array)
    except Exception as e:
        print(f"Warning: Failed to save segmented LBP cache for {image_name}: {e}")
    
    return features_array


def compute_lbp_for_segments_by_categories(
    categories: list[str],
    segmented_images: Dict[str, Dict[str, np.ndarray]],
    radius: int = 2,
    n_points: int = 8,
    method: str = "nri_uniform",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute LBP for segmented images organized by category.
    
    Args:
        categories: List of category names
        segmented_images: Dict structure {category: {image_name: regions_matrix}}
        radius: LBP radius parameter
        n_points: LBP n_points parameter
        method: LBP method
        
    Returns:
        Dict structure {category: {image_name: features_array}}
    """
    return {
        category: compute_lbp_for_many_segmented_images(
            segmented_images[category], radius, n_points, method
        )
        for category in categories
    }


def compute_lbp_for_many_segmented_images(
    images: Dict[str, np.ndarray],
    radius: int,
    n_points: int,
    method: str = "nri_uniform",
) -> Dict[str, np.ndarray]:
    """
    Compute LBP for all segmented images in a single category.
    
    Args:
        category_segmented_images: Dict of {image_name: regions_matrix}
        radius: LBP radius parameter
        n_points: LBP n_points parameter
        method: LBP method
        
    Returns:
        Dict structure {image_name: features_array}
    """
    lbps = {}
    
    for image_name, regions_matrix in images.items():
        lbps[image_name] = compute_lbp_for_segmented_image(
            image_name, regions_matrix, radius, n_points, method
        )
    
    return lbps


def _is_valid_region(region) -> bool:
    """
    Check if a region is valid for LBP computation.
    
    Args:
        region: Region data to validate
        
    Returns:
        bool: True if region is valid, False otherwise
    """
    return (
        region is not None 
        and hasattr(region, 'shape') 
        and region.size > 0
        and len(region.shape) == 2  # Ensure it's a 2D array
        and region.shape[0] > 0 
        and region.shape[1] > 0
    )


def build_histogram_from_lbp(lbp, n_pixels):
    n_bins = int(lbp.max() + 1)
    histogram, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # normalize the histogram
    histogram = histogram.astype("float")
    histogram /= histogram.sum() + 1e-6

    return histogram


def build_histogram_from_many_lbps(
    lbps: Dict[str, np.ndarray], n_pixels: int
) -> Dict[str, np.ndarray]:
    return {name: build_histogram_from_lbp(lbp, n_pixels) for name, lbp in lbps.items()}


def build_histograms_from_categories(
    categories: list[str], lbps: Dict[str, Dict[str, np.ndarray]], n_pixels: int
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        category: build_histogram_from_many_lbps(lbps[category], n_pixels)
        for category in categories
    }
