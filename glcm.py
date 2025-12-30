from skimage.feature import graycomatrix, graycoprops
import numpy as np
from joblib import Parallel, delayed

from typing import Dict
from pathlib import Path


def calculate_glcm_from_single_image(
    image, distances, angles, levels, symmetric=True, normed=True
):
    return graycomatrix(
        image, distances, angles, levels, symmetric=symmetric, normed=normed
    )


def calculate_glcm_from_many_images(
    images, distances, angles, levels, symmetric=True, normed=True
):
    return [
        calculate_glcm_from_single_image(
            img, distances, angles, levels, symmetric, normed
        )
        for img in images
    ]


def calculate_glcm_matrix_for_each_category(
    categories, images, distances, angles, levels, symmetric=True, normed=True
):
    return {
        category: calculate_glcm_from_many_images(
            images[category], distances, angles, levels, symmetric, normed
        )
        for category in categories
    }


def extract_glcm_features(glcm, props):
    return np.array([graycoprops(glcm, prop).flatten() for prop in props]).flatten()


def extract_glcm_features_from_many_images(glcm_list, props):
    return [extract_glcm_features(glcm, props) for glcm in glcm_list]


def extract_glcm_features_for_each_category(categories, glcm_list, props):
    return {
        category: extract_glcm_features_from_many_images(glcm_list[category], props)
        for category in categories
    }


# Nova função para paralelizar a extração de características GLCM
def parallel_extract_glcm_features_from_many_images(glcm_list, props, n_jobs=-1):
    return Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(extract_glcm_features)(glcm, props) for glcm in glcm_list
    )


def parallel_extract_glcm_features_for_each_category(
    categories: Dict[str, Dict[str, np.ndarray]], glcm_list, props, n_jobs=-1
):
    return {
        category: parallel_extract_glcm_features_from_many_images(
            glcm_list[category], props, n_jobs
        )
        for category in categories
    }


# --- Full-image GLCM extraction with caching (analogous to LBP flow) ---
def calculate_glcm_for_single_image(
    image_name: str,
    image_value: np.ndarray,
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
):
    """
    Calculate GLCM features for a full image (not segmented) with caching support.

    Saves cached features to features/glcms/<image_name>_glcm.npy and returns the
    extracted GLCM properties as a 1D numpy array.
    """
    cache_dir = Path("features/glcms")
    cache_file = cache_dir / f"{image_name}_glcm.npy"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        try:
            features = np.load(cache_file)
            return features
        except Exception as e:
            print(f"Warning: Failed to load cached GLCM for {image_name}: {e}")

    # compute matrix and extract features
    glcm_matrix = graycomatrix(
        image_value, distances, angles, levels, symmetric=symmetric, normed=normed
    )
    features = extract_glcm_features(glcm_matrix, glcm_props)

    try:
        np.save(cache_file, features)
    except Exception as e:
        print(f"Warning: Failed to save GLCM cache for {image_name}: {e}")

    return features


def parallel_calculate_glcm_from_many_images_full(
    images: Dict[str, np.ndarray],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
    n_jobs=-1,
) -> Dict[str, np.ndarray]:
    """
    Parallel runner that computes cached full-image GLCM features for many images.

    Returns a dict {image_name: features_array}.
    """
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(calculate_glcm_for_single_image)(
            name, img, distances, angles, glcm_props, levels, symmetric, normed
        )
        for name, img in images.items()
    )

    image_names = list(images.keys())
    return {image_names[i]: results[i] for i in range(len(results))}


def calculate_glcm_from_segmented_image(
    image_name: str,
    regions_matrix: np.ndarray,
    distances: list[int],
    glcm_props: list[str],
    angles: np.ndarray,
    levels: int,
    symmetric=True,
    normed=True,
) -> np.ndarray:
    """
    Calculate GLCM features for all segments of a single image with caching support.

    Args:
        image_name (str): Name of the image file (without extension)
        regions_matrix (np.ndarray): Matrix containing image segments
        distances (list[int]): List of distances for GLCM computation
        angles (np.ndarray): Array of angles for GLCM computation
        levels (int): Number of gray levels for GLCM computation
        symmetric (bool): Whether to use symmetric GLCM
        normed (bool): Whether to normalize GLCM

    Returns:
        np.ndarray: Array containing GLCM matrices for all segments
    """

    # Define cache directory and file path
    cache_dir = Path("features/glcms")
    cache_file = cache_dir / f"{image_name}_segmented_glcm.npy"

    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Try to load from cache first
    if cache_file.exists():
        try:
            features_array = np.load(cache_file)
            return features_array
        except Exception as e:
            print(
                f"Warning: Failed to load cached segmented GLCM for {image_name}: {e}"
            )
            # Continue to compute GLCM if loading fails

    # Compute GLCM for all segments
    features_list = []
    rows, cols = regions_matrix.shape

    for row in range(rows):
        for col in range(cols):
            region = regions_matrix[row, col]
            if _is_valid_region(region):
                try:
                    glcm_matrix = graycomatrix(
                        region,
                        distances,
                        angles,
                        levels,
                        symmetric=symmetric,
                        normed=normed,
                    )
                    glcm_features = extract_glcm_features(glcm_matrix, glcm_props)
                    features_list.append(glcm_features)
                except Exception as e:
                    print(
                        f"Warning: Failed to compute GLCM for {image_name} region ({row},{col}): {e}"
                    )
                    # Add zero array as placeholder for failed regions (expecting 72 features)
                    expected_features = 72  # Based on your configuration
                    features_list.append(np.zeros(expected_features))

    # Convert to numpy array with padding if necessary
    if features_list:
        # Check if all feature arrays have the same shape
        if len(features_list) > 1:
            shapes = [len(features) for features in features_list]
            max_shape = max(shapes)

            # Pad feature arrays to have the same size
            normalized_features = []
            for features in features_list:
                if len(features) < max_shape:
                    # Pad with zeros
                    padded_features = np.zeros(max_shape)
                    padded_features[: len(features)] = features
                    normalized_features.append(padded_features)
                else:
                    normalized_features.append(features)

            features_array = np.array(normalized_features)
        else:
            features_array = np.array(features_list)
    else:
        # Fallback for completely empty image
        expected_features = 72
        features_array = np.array([np.zeros(expected_features)])

    # Save to cache
    try:
        np.save(cache_file, features_array)
    except Exception as e:
        print(f"Warning: Failed to save segmented GLCM cache for {image_name}: {e}")

    return features_array


def _is_valid_region(region) -> bool:
    """
    Check if a region is valid for GLCM computation.

    Args:
        region: Region data to validate

    Returns:
        bool: True if region is valid, False otherwise
    """
    return (
        region is not None
        and hasattr(region, "shape")
        and region.size > 0
        and len(region.shape) == 2  # Ensure it's a 2D array
        and region.shape[0] > 0
        and region.shape[1] > 0
    )


def parallel_calculate_glcm_from_many_images_segmented(
    images: Dict[str, np.ndarray],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
    n_jobs=-1,
) -> Dict[str, np.ndarray]:
    # Execute parallel computation
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(calculate_glcm_from_segmented_image)(
            key,
            regions_matrix,
            distances,
            glcm_props,
            angles,
            levels,
            symmetric,
            normed,
        )
        for key, regions_matrix in images.items()
    )

    # Convert list of results back to dictionary format
    image_names = list(images.keys())
    return {image_names[i]: results[i] for i in range(len(results))}


def parallel_calculate_glcm_for_each_category_segmented(
    categories: list[str],
    images: Dict[str, Dict[str, np.ndarray]],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
    n_jobs=-1,
) -> Dict[str, Dict[str, np.ndarray]]:
    return {
        category: parallel_calculate_glcm_from_many_images_segmented(
            images[category],
            distances,
            angles,
            glcm_props,
            levels,
            symmetric,
            normed,
            n_jobs,
        )
        for category in categories
    }


def calculate_glcm_for_many_segmented_images(
    images: Dict[str, np.ndarray],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
) -> Dict[str, np.ndarray]:
    """
    Calculate GLCM for all segmented images in a single category.

    Args:
        images: Dict of {image_name: regions_matrix}
        distances: List of distances for GLCM computation
        angles: Array of angles for GLCM computation
        glcm_props: List of GLCM properties to extract
        levels: Number of gray levels for GLCM computation
        symmetric: Whether to use symmetric GLCM
        normed: Whether to normalize GLCM

    Returns:
        Dict structure {image_name: features_array}
    """
    glcms = {}

    for image_name, regions_matrix in images.items():
        glcms[image_name] = calculate_glcm_from_segmented_image(
            image_name,
            regions_matrix,
            distances,
            glcm_props,
            angles,
            levels,
            symmetric,
            normed,
        )

    return glcms


def calculate_glcm_for_segments_by_categories(
    categories: list[str],
    segmented_images: Dict[str, Dict[str, np.ndarray]],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate GLCM for segmented images organized by category.

    Args:
        categories: List of category names
        segmented_images: Dict structure {category: {image_name: regions_matrix}}
        distances: List of distances for GLCM computation
        angles: Array of angles for GLCM computation
        glcm_props: List of GLCM properties to extract
        levels: Number of gray levels for GLCM computation
        symmetric: Whether to use symmetric GLCM
        normed: Whether to normalize GLCM

    Returns:
        Dict structure {category: {image_name: features_array}}
    """
    return {
        category: calculate_glcm_for_many_segmented_images(
            segmented_images[category],
            distances,
            angles,
            glcm_props,
            levels,
            symmetric,
            normed,
        )
        for category in categories
    }

def parallel_calculate_glcm_for_each_category(
    categories: list[str],
    images: Dict[str, Dict[str, np.ndarray]],
    distances: list[int],
    angles: np.ndarray,
    glcm_props: list[str],
    levels: int,
    symmetric=True,
    normed=True,
    n_jobs=-1,
) -> Dict[str, Dict[str, np.ndarray]]:
    # For full-image extraction and caching use the full-image parallel runner.
    return {
        category: parallel_calculate_glcm_from_many_images_full(
            images[category],
            distances,
            angles,
            glcm_props,
            levels,
            symmetric,
            normed,
            n_jobs,
        )
        for category in categories
    }
