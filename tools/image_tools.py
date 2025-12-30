"""
Image Tools Module

This module provides utilities for image processing, specifically for segmenting images
into regions based on a segmentation factor.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Dict, Any
import os

import skimage as ski
from skimage.color import rgb2gray, rgba2rgb
from skimage import transform


def segment_image_into_grid(image: np.ndarray, segmentation_factor: int) -> np.ndarray:
    """
    Segments an image into a grid of regions based on the segmentation factor.

    Args:
        image (np.ndarray): Input grayscale image as numpy array
        segmentation_factor (int): Number of segments per dimension (creates segmentation_factor x segmentation_factor grid)

    Returns:
        np.ndarray: A segmentation_factor x segmentation_factor matrix where each element
                   contains the pixel values of that region as a numpy array
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    height, width = image.shape

    # Calculate the size of each region
    region_height = height // segmentation_factor
    region_width = width // segmentation_factor

    # Create a matrix to hold the regions
    regions_matrix = np.empty((segmentation_factor, segmentation_factor), dtype=object)

    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            # Calculate coordinates for this region
            start_row = row_idx * region_height
            end_row = start_row + region_height
            start_col = col_idx * region_width
            end_col = start_col + region_width

            # Handle edge cases for the last regions if image dimensions aren't perfectly divisible
            if row_idx == segmentation_factor - 1:
                end_row = height
            if col_idx == segmentation_factor - 1:
                end_col = width

            # Extract the region and store in matrix
            region = image[start_row:end_row, start_col:end_col]
            regions_matrix[row_idx, col_idx] = region

    return regions_matrix


def calculate_optimal_grid(
    height: int, width: int, k: int, min_region_size: int = 64
) -> Tuple[int, int, int]:
    """
    Calculate the optimal grid dimensions that best preserve image proportions.

    Args:
        height (int): Image height
        width (int): Image width
        k (int): Desired number of regions
        min_region_size (int): Minimum size for each region dimension

    Returns:
        Tuple[int, int, int]: (rows, cols, actual_regions_count)
    """
    aspect_ratio = width / height
    best_option = None
    best_score = float("inf")

    # Find all possible factorizations of k and smaller values
    max_possible_k = min(k, (height // min_region_size) * (width // min_region_size))

    for target_k in range(max_possible_k, 0, -1):
        # Find all divisor pairs for target_k
        for rows in range(1, int(math.sqrt(target_k)) + 1):
            if target_k % rows == 0:
                cols = target_k // rows

                # Check if regions meet minimum size requirement
                region_height = height // rows
                region_width = width // cols

                if region_height >= min_region_size and region_width >= min_region_size:
                    # Calculate how well this preserves the aspect ratio
                    grid_ratio = cols / rows
                    proportion_score = abs(grid_ratio - aspect_ratio)

                    # Prefer solutions closer to target k
                    k_penalty = (k - target_k) * 0.1

                    total_score = proportion_score + k_penalty

                    if total_score < best_score:
                        best_score = total_score
                        best_option = (rows, cols, target_k)

                # Also try the reversed option (cols, rows)
                if rows != cols:  # Avoid duplicates for square grids
                    region_height = height // cols
                    region_width = width // rows

                    if (
                        region_height >= min_region_size
                        and region_width >= min_region_size
                    ):
                        grid_ratio = rows / cols
                        proportion_score = abs(grid_ratio - aspect_ratio)
                        k_penalty = (k - target_k) * 0.1
                        total_score = proportion_score + k_penalty

                        if total_score < best_score:
                            best_score = total_score
                            best_option = (cols, rows, target_k)

    if best_option is None:
        # Fallback: single region if nothing fits
        return (1, 1, 1)

    return best_option


def segment_image_dynamic(
    image: np.ndarray, k: int, min_region_size: int = 64
) -> np.ndarray:
    """
    Dynamically segments an image into regions based on total count and minimum size.

    Args:
        image (np.ndarray): Input grayscale image as numpy array
        k (int): Desired total number of regions
        min_region_size (int): Minimum size for each region dimension

    Returns:
        np.ndarray: A rows x cols matrix where each element contains the pixel values
                   of that region as a numpy array
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    height, width = image.shape

    # Calculate optimal grid
    rows, cols, actual_k = calculate_optimal_grid(height, width, k, min_region_size)

    # Calculate the size of each region
    region_height = height // rows
    region_width = width // cols

    # Create a matrix to hold the regions
    regions_matrix = np.empty((rows, cols), dtype=object)

    for row_idx in range(rows):
        for col_idx in range(cols):
            # Calculate coordinates for this region
            start_row = row_idx * region_height
            end_row = start_row + region_height
            start_col = col_idx * region_width
            end_col = start_col + region_width

            # Handle edge cases for the last regions
            if row_idx == rows - 1:
                end_row = height
            if col_idx == cols - 1:
                end_col = width

            # Extract the region and store in matrix
            region = image[start_row:end_row, start_col:end_col]
            regions_matrix[row_idx, col_idx] = region

    return regions_matrix


def segment_images_batch(
    images: List[np.ndarray], segmentation_factor: int
) -> List[np.ndarray]:
    """
    Segments a batch of images into regions.

    Args:
        images (List[np.ndarray]): List of grayscale images
        segmentation_factor (int): Number of segments per dimension

    Returns:
        List[np.ndarray]: List of segmentation matrices, one for each input image.
                         Each matrix is segmentation_factor x segmentation_factor with
                         pixel arrays in each cell.
    """
    all_segmented_images = []

    for i, image in enumerate(images):
        try:
            segmented_regions = segment_image_into_grid(image, segmentation_factor)
            all_segmented_images.append(segmented_regions)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # Create empty matrix for failed images
            empty_matrix = np.empty(
                (segmentation_factor, segmentation_factor), dtype=object
            )
            all_segmented_images.append(empty_matrix)

    return all_segmented_images


def segment_images_by_category(
    images_by_category: Dict[str, List[np.ndarray]], segmentation_factor: int
) -> Dict[str, List[np.ndarray]]:
    """
    Segments images organized by category.

    Args:
        images_by_category (Dict[str, List[np.ndarray]]): Dictionary with category names as keys
                                                         and lists of images as values
        segmentation_factor (int): Number of segments per dimension

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary with category names as keys and
                                   lists of segmentation matrices as values
    """
    segmented_by_category = {}

    for category, images in images_by_category.items():
        print(f"Segmenting {len(images)} images from category '{category}'...")
        segmented_images = segment_images_batch(images, segmentation_factor)
        segmented_by_category[category] = segmented_images

        # Print summary
        total_regions = (
            len(segmented_images) * segmentation_factor * segmentation_factor
        )
        print(
            f"  Generated {total_regions} regions ({segmentation_factor}x{segmentation_factor} grid per image)"
        )

    return segmented_by_category


def visualize_image_segmentation(
    image: np.ndarray, segmentation_factor: int, figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualizes the original image and its segmented regions.

    Args:
        image (np.ndarray): Input grayscale image
        segmentation_factor (int): Number of segments per dimension
        figsize (Tuple[int, int]): Figure size for matplotlib
    """
    regions_matrix = segment_image_into_grid(image, segmentation_factor)

    # Calculate subplot grid
    cols = segmentation_factor
    rows = segmentation_factor

    plt.figure(figsize=figsize)

    # Show each region
    plot_idx = 1
    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(regions_matrix[row_idx, col_idx], cmap="gray")
            plt.title(f"Region ({row_idx},{col_idx})")
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()


def get_region_statistics(regions_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for a matrix of image regions.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions (segmentation_factor x segmentation_factor)

    Returns:
        Dict[str, Any]: Statistics including min/max/mean dimensions and total regions
    """
    # Check if the matrix is empty or has no dimensions
    if regions_matrix.shape == (0,) or len(regions_matrix.shape) == 0:
        return {}

    # Get all region shapes
    region_shapes = []
    total_regions = 0

    for row_idx in range(regions_matrix.shape[0]):
        for col_idx in range(regions_matrix.shape[1]):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, "shape"):
                region_shapes.append(region.shape)
                total_regions += 1

    if not region_shapes:
        return {}

    heights = [shape[0] for shape in region_shapes]
    widths = [shape[1] for shape in region_shapes]

    stats = {
        "total_regions": total_regions,
        "grid_shape": regions_matrix.shape,
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": np.mean(heights),
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": np.mean(widths),
    }

    return stats


def reconstruct_image_from_regions(regions_matrix: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from its segmented regions matrix.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions (segmentation_factor x segmentation_factor)

    Returns:
        np.ndarray: Reconstructed image
    """
    # Check if the matrix is empty or has no dimensions
    if regions_matrix.shape == (0,) or len(regions_matrix.shape) == 0:
        raise ValueError("No regions provided for reconstruction")

    rows, cols = regions_matrix.shape

    # Get the first non-None region to determine data type
    first_region = None
    for row_idx in range(rows):
        for col_idx in range(cols):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, "shape"):
                first_region = region
                break
        if first_region is not None:
            break

    if first_region is None:
        raise ValueError("All regions are None")

    # Calculate total image dimensions by examining actual region sizes
    # This handles dynamic segmentation where edge regions may have different sizes
    total_height = 0
    total_width = 0
    
    # Calculate height by checking the first column
    for row_idx in range(rows):
        region = regions_matrix[row_idx, 0]
        if region is not None and hasattr(region, "shape"):
            total_height += region.shape[0]
    
    # Calculate width by checking the first row
    for col_idx in range(cols):
        region = regions_matrix[0, col_idx]
        if region is not None and hasattr(region, "shape"):
            total_width += region.shape[1]

    # Create the reconstructed image
    reconstructed = np.zeros((total_height, total_width), dtype=first_region.dtype)

    # Fill in each region using actual positions
    current_row = 0
    for row_idx in range(rows):
        current_col = 0
        row_height = 0
        
        for col_idx in range(cols):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, "shape"):
                region_height, region_width = region.shape
                
                # Place the region at its correct position
                reconstructed[current_row:current_row + region_height, 
                            current_col:current_col + region_width] = region
                
                current_col += region_width
                row_height = max(row_height, region_height)
        
        current_row += row_height

    return reconstructed


def segment_images_batch_dynamic(
    images: List[np.ndarray], k: int, min_region_size: int = 64
) -> List[np.ndarray]:
    """
    Dynamically segments a batch of images into regions.

    Args:
        images (List[np.ndarray]): List of grayscale images
        k (int): Desired total number of regions per image
        min_region_size (int): Minimum size for each region dimension

    Returns:
        List[np.ndarray]: List of segmentation matrices, one for each input image.
                         Each matrix has dimensions based on optimal grid calculation.
    """
    all_segmented_images = []

    for i, image in enumerate(images):
        try:
            segmented_regions = segment_image_dynamic(image, k, min_region_size)
            all_segmented_images.append(segmented_regions)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # Create single region fallback for failed images
            single_region_matrix = np.empty((1, 1), dtype=object)
            single_region_matrix[0, 0] = image
            all_segmented_images.append(single_region_matrix)

    return all_segmented_images


def segment_images_by_category_dynamic(
    images_by_category: Dict[str, List[np.ndarray]], k: int, min_region_size: int = 64
) -> Dict[str, List[np.ndarray]]:
    """
    Dynamically segments images organized by category.

    Args:
        images_by_category (Dict[str, List[np.ndarray]]): Dictionary with category names as keys
                                                         and lists of images as values
        k (int): Desired total number of regions per image
        min_region_size (int): Minimum size for each region dimension

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary with category names as keys and
                                   lists of segmentation matrices as values
    """
    segmented_by_category = {}

    for category, images in images_by_category.items():
        print(
            f"Dynamically segmenting {len(images)} images from category '{category}' (target: {k} regions)..."
        )
        segmented_images = segment_images_batch_dynamic(images, k, min_region_size)
        segmented_by_category[category] = segmented_images

        # Calculate actual regions generated (may vary per image due to different dimensions)
        total_regions = 0
        grid_info = {}

        for seg_matrix in segmented_images:
            actual_regions = seg_matrix.shape[0] * seg_matrix.shape[1]
            total_regions += actual_regions
            grid_key = f"{seg_matrix.shape[0]}x{seg_matrix.shape[1]}"
            grid_info[grid_key] = grid_info.get(grid_key, 0) + 1

        avg_regions = total_regions / len(segmented_images) if segmented_images else 0
        print(f"  Generated average {avg_regions:.1f} regions per image")

        # Show grid distribution
        for grid, count in grid_info.items():
            print(f"    {count} images with {grid} grid")

    return segmented_by_category


def visualize_image_segmentation_dynamic(
    image: np.ndarray,
    k: int,
    min_region_size: int = 64,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Visualizes an image and its dynamically segmented regions.

    Args:
        image (np.ndarray): Input grayscale image
        k (int): Desired total number of regions
        min_region_size (int): Minimum size for each region dimension
        figsize (Tuple[int, int]): Figure size for matplotlib
    """
    regions_matrix = segment_image_dynamic(image, k, min_region_size)

    rows, cols = regions_matrix.shape

    plt.figure(figsize=figsize)

    # Show each region
    plot_idx = 1
    for row_idx in range(rows):
        for col_idx in range(cols):
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(regions_matrix[row_idx, col_idx], cmap="gray")
            plt.title(f"Region ({row_idx},{col_idx})")
            plt.axis("off")
            plot_idx += 1

    plt.suptitle(f"Dynamic Segmentation: {rows}x{cols} grid ({rows * cols} regions)")
    plt.tight_layout()
    plt.show()


def calculate_dynamic_k(
    height: int,
    width: int,
    min_k: int = 4,
    max_k: int = 20,
    base_size: int = 512,
    base_k: int = 9,
) -> int:
    """
    Calculate K dynamically based on image dimensions.

    Uses 512x512 with K=9 as reference point and scales proportionally.

    Args:
        height (int): Image height
        width (int): Image width
        min_k (int): Minimum K value (default: 4)
        max_k (int): Maximum K value (default: 49)
        base_size (int): Reference image size (default: 512)
        base_k (int): Reference K for base_size image (default: 9)

    Returns:
        int: Calculated K value, constrained between min_k and max_k
    """
    # Calculate total area relative to base size
    image_area = height * width
    base_area = base_size * base_size
    area_ratio = image_area / base_area

    # Scale K proportionally to area
    # For 512x512 (area_ratio=1.0), we want K=9
    # For larger images, K increases proportionally
    calculated_k = int(base_k * area_ratio)

    # Apply min/max constraints
    k = max(min_k, min(calculated_k, max_k))

    # Ensure K is reasonable (prefer perfect squares or near-perfect rectangles)
    perfect_squares = [4, 9, 16, 25, 36, 49, 64, 81, 100]
    good_rectangles = [6, 8, 10, 12, 15, 18, 20, 24, 28, 30, 32, 35, 40, 42, 45, 48]

    valid_k_values = [
        k_val for k_val in perfect_squares + good_rectangles if min_k <= k_val <= max_k
    ]

    if k in valid_k_values:
        return k

    # Find closest valid K
    return min(valid_k_values, key=lambda x: abs(x - k))


def segment_image_auto(
    image: np.ndarray,
    min_region_size: int = 128,
    min_k: int = 4,
    max_k: int = 20,
    base_size: int = 512,
    base_k: int = 9,
) -> np.ndarray:
    """
    Automatically segments an image by dynamically calculating K.

    Args:
        image (np.ndarray): Input grayscale image as numpy array
        min_region_size (int): Minimum size for each region dimension
        min_k (int): Minimum K value
        max_k (int): Maximum K value
        base_size (int): Reference image size
        base_k (int): Reference K for base_size image

    Returns:
        np.ndarray: A rows x cols matrix where each element contains the pixel values
                   of that region as a numpy array
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    height, width = image.shape

    # Calculate K dynamically
    k = calculate_dynamic_k(height, width, min_k, max_k, base_size, base_k)

    # Use dynamic segmentation with calculated K
    return segment_image_dynamic(image, k, min_region_size)


def segment_images_batch_auto(
    images: Dict[str, np.ndarray],
    min_region_size: int = 128,
    min_k: int = 4,
    max_k: int = 20,
    base_size: int = 512,
    base_k: int = 9,
) -> Tuple[Dict[str, np.ndarray], int, float]:
    """
    Automatically segments a batch of images with dynamic K calculation.

    Args:
        images (Dict[str, np.ndarray]): Dictionary of grayscale images
        min_region_size (int): Minimum size for each region dimension
        min_k (int): Minimum K value
        max_k (int): Maximum K value
        base_size (int): Reference image size
        base_k (int): Reference K for base_size image

    Returns:
        (Dict[str, np.ndarray], int, float): Tuple containing:
            - Dictionary of segmentation matrices for each image
            - Total number of regions generated across all images
            - Average number of regions per image
    """
    all_segmented_images = {}
    total_regions = 0

    for key, image in images.items():
        try:
            segmented_regions = segment_image_auto(
                image, min_region_size, min_k, max_k, base_size, base_k
            )
            all_segmented_images[key] = segmented_regions
            total_regions += segmented_regions.shape[0] * segmented_regions.shape[1]
        except Exception as e:
            print(f"Error processing image {key}: {e}")
            # Create single region fallback for failed images
            single_region_matrix = np.empty((1, 1), dtype=object)
            single_region_matrix[0, 0] = image
            all_segmented_images[key] = single_region_matrix

    k_mean = total_regions / len(images) if images else 0
    return (all_segmented_images, total_regions, k_mean)


def segment_images_by_category_auto(
    images_by_category: Dict[str, Dict[str, np.ndarray]],
    min_region_size: int = 128,
    min_k: int = 4,
    max_k: int = 20,
    base_size: int = 512,
    base_k: int = 9,
) -> Dict[str, Dict[str, np.ndarray]:]:
    """
    Automatically segments images organized by category with dynamic K calculation.

    Args:
        images_by_category (Dict[str, List[np.ndarray]]): Dictionary with category names as keys
        min_region_size (int): Minimum size for each region dimension
        min_k (int): Minimum K value
        max_k (int): Maximum K value
        base_size (int): Reference image size
        base_k (int): Reference K for base_size image

    Returns:
        Dict[str, Dict[str, np.ndarray] :]: Dictionary with category names as keys and                                    dictionaries of segmentation matrices as values
    """
    segmented_by_category = {}

    for category, images in images_by_category.items():
        print(
            f"Auto-segmenting {len(images)} images from category '{category}' (K range: {min_k}-{max_k})..."
        )
        (segmented_images, total_regions, k_mean) = segment_images_batch_auto(
            images, min_region_size, min_k, max_k, base_size, base_k
        )
        segmented_by_category[category] = segmented_images

        print(f"  Generated average {k_mean:.1f} regions per image")
        print(f"{category}:", total_regions, "regions total")

    return segmented_by_category

def merge_image_categories_dicts(images_by_categories: Dict[str, Dict[str, np.ndarray]:]):
    """
    Merges multiple category dictionaries into a single dictionary.

    Args:
        images_by_categories (Dict[str, Dict[str, np.ndarray]]): Dictionary with category names as keys
                                                                 and dictionaries of images as values

    Returns:
        Dict[str, np.ndarray]: Merged dictionary with all images
    """
    merged_dict = {}
    for category_dict in images_by_categories.values():
        merged_dict.update(category_dict)
    return merged_dict


def visualize_image_segmentation_auto(
    image: np.ndarray,
    min_region_size: int = 128,
    min_k: int = 4,
    max_k: int = 20,
    base_size: int = 512,
    base_k: int = 9,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Visualizes an image and its automatically segmented regions.

    Args:
        image (np.ndarray): Input grayscale image
        min_region_size (int): Minimum size for each region dimension
        min_k (int): Minimum K value
        max_k (int): Maximum K value
        base_size (int): Reference image size
        base_k (int): Reference K for base_size image
        figsize (Tuple[int, int]): Figure size for matplotlib
    """
    height, width = image.shape
    k = calculate_dynamic_k(height, width, min_k, max_k, base_size, base_k)

    regions_matrix = segment_image_auto(
        image, min_region_size, min_k, max_k, base_size, base_k
    )

    rows, cols = regions_matrix.shape
    actual_k = rows * cols

    plt.figure(figsize=figsize)

    # Show each region
    plot_idx = 1
    for row_idx in range(rows):
        for col_idx in range(cols):
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(regions_matrix[row_idx, col_idx], cmap="gray")
            plt.title(f"Region ({row_idx},{col_idx})")
            plt.axis("off")
            plot_idx += 1

    plt.suptitle(
        f"Auto Segmentation: {height}x{width} → {rows}x{cols} grid (K={actual_k}, target K={k})"
    )
    plt.tight_layout()
    plt.show()


def get_region(regions_matrix: np.ndarray, row: int, col: int) -> np.ndarray:
    """
    Get a specific region from the segmentation matrix.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions
        row (int): Row index of the region
        col (int): Column index of the region

    Returns:
        np.ndarray: The region's pixel data
    """
    return regions_matrix[row, col]


def show_region_example(regions_matrix: np.ndarray, row: int = 0, col: int = 0):
    """
    Display a specific region from the segmentation matrix.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions
        row (int): Row index of the region to show
        col (int): Column index of the region to show
    """
    region = get_region(regions_matrix, row, col)

    plt.figure(figsize=(6, 6))
    plt.imshow(region, cmap="gray")
    plt.title(f"Region ({row}, {col}) - Shape: {region.shape}")
    plt.axis("off")
    plt.show()


def resize_with_padding(img, target_size=512):
    h, w = img.shape[:2]

    # escala baseada na maior dimensão
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # resize proporcional
    resized = transform.resize(
        img, (new_h, new_w), anti_aliasing=True, preserve_range=True
    )

    # cria canvas preto
    if img.ndim == 3:
        new_img = np.zeros(
            (target_size, target_size, img.shape[2]), dtype=resized.dtype
        )
    else:  # grayscale
        new_img = np.zeros((target_size, target_size), dtype=resized.dtype)

    # centraliza a imagem redimensionada dentro do canvas
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    new_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return new_img


def load_images_from_category(
    directory: str, category: str, allowed_extensions: List[str], resize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Load images from a specific category directory and return as dictionary.

    Args:
        directory (str): Base directory path containing category subdirectories
        category (str): Category subdirectory name
        allowed_extensions (List[str]): List of allowed file extensions (e.g., ['.jpg', '.png'])
        resize (bool): Whether to resize images with padding to standard size

    Returns:
        Dict[str, np.ndarray]: Dictionary with filename (without extension) as key
                              and processed grayscale image as value
    """
    import glob

    # Coletar todos os arquivos de imagem com seus caminhos
    all_files = []
    for ext in allowed_extensions:
        pattern = os.path.join(directory, category, f"*{ext}")
        all_files.extend(glob.glob(pattern))

    # Ordenar para garantir consistência
    all_files.sort()

    # Criar dicionário com nome do arquivo como chave
    images_dict = {}

    for file_path in all_files:
        # Extrair nome do arquivo sem extensão
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Carregar e processar a imagem
        img = ski.io.imread(file_path)
        # Se a imagem veio com canal alpha (RGBA), converte para RGB antes de converter para cinza
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)
        gray_img = rgb2gray(img)

        if resize:
            gray_img = resize_with_padding(gray_img)

        gray_img_uint8 = ski.util.img_as_ubyte(gray_img)
        images_dict[file_name] = gray_img_uint8

    return images_dict


def load_train_images_dict(
    directory: str,
    categories: List[str],
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png"],
    resize: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load images from multiple categories and return nested dictionary structure.

    Args:
        directory (str): Base directory path containing category subdirectories
        categories (List[str]): List of category names (subdirectory names)
        allowed_extensions (List[str]): List of allowed file extensions
        resize (bool): Whether to resize images with padding to standard size

    Returns:
        Dict[str, Dict[str, np.ndarray]]: Nested dictionary structure:
                                        {category: {filename: image_array}}
    """
    return {
        category: load_images_from_category(
            directory, category, allowed_extensions, resize
        )
        for category in categories
    }
