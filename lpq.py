import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

from typing import Dict, List

from pathlib import Path


def lpq(img: np.ndarray, winSize=7, decorr=1, mode="nh") -> np.ndarray:
    """
    Local Phase Quantization (LPQ) texture descriptor implementation.
    
    LPQ is a texture descriptor that extracts local phase information using
    Short-Time Fourier Transform (STFT) in a local window around each pixel.
    
    Args:
        img (np.ndarray): Input grayscale image (2D array)
        winSize (int): Size of the local window (default: 7)
        decorr (int): Apply decorrelation (1) or not (0) (default: 1)
        mode (str): Output mode:
                   - "nh": Normalized histogram (default)
                   - "h": Histogram
                   - "im": LPQ code image
                   
    Returns:
        np.ndarray: LPQ descriptor
                   - For mode "nh"/"h": 256-element histogram
                   - For mode "im": LPQ code image with same spatial dimensions
                   
    Note:
        The LPQ descriptor is computed by:
        1. Computing STFT responses at 4 frequency points
        2. Optional decorrelation using whitening transform
        3. Quantizing phase information into 8-bit codes
        4. Computing histogram of quantized codes (for histogram modes)
    """
    rho = 0.90

    STFTalpha = (
        1 / winSize
    )  # alpha in STFT approaches (for Gaussian derivative alpha=1)

    convmode = "valid"  # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img = np.float64(img)  # Convert np.image to double
    r = (winSize - 1) / 2  # Get radius from window size
    x = np.arange(-r, r + 1)[np.newaxis]  # Form spatial coordinates in window

    #  STFT uniform window
    #  Basic STFT filters
    w0 = np.ones_like(x)
    w1 = np.exp(-2 * np.pi * x * STFTalpha * 1j)
    w2 = np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1 = convolve2d(convolve2d(img, w0.T, convmode), w1, convmode)
    filterResp2 = convolve2d(convolve2d(img, w1.T, convmode), w0, convmode)
    filterResp3 = convolve2d(convolve2d(img, w1.T, convmode), w1, convmode)
    filterResp4 = convolve2d(convolve2d(img, w1.T, convmode), w2, convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp = np.dstack(
        [
            filterResp1.real,
            filterResp1.imag,
            filterResp2.real,
            filterResp2.imag,
            filterResp3.real,
            filterResp3.imag,
            filterResp4.real,
            filterResp4.imag,
        ]
    )

    if decorr == 1:
        xp, yp = np.meshgrid(np.arange(1, winSize + 1), np.arange(1, winSize + 1))
        pp = np.column_stack((yp.flatten(), xp.flatten()))
        dd = cdist(pp, pp)
        C = rho**dd

        q1 = w0.reshape((winSize, 1)) @ w1.reshape((1, winSize))
        q2 = w1.reshape((winSize, 1)) @ w0.reshape((1, winSize))
        q3 = w1.reshape((winSize, 1)) @ w1.reshape((1, winSize))
        q4 = w1.reshape((winSize, 1)) @ w2.reshape((1, winSize))

        M = np.vstack(
            (
                q1.real.T.ravel(),
                q1.imag.T.ravel(),
                q2.real.T.ravel(),
                q2.imag.T.ravel(),
                q3.real.T.ravel(),
                q3.imag.T.ravel(),
                q4.real.T.ravel(),
                q4.imag.T.ravel(),
            )
        )

        D = np.dot(M, C).dot(M.T)
        A = np.diag(
            [1.000007, 1.000006, 1.000005, 1.000004, 1.000003, 1.000002, 1.000001, 1]
        )
        U, S, V = np.linalg.svd(np.dot(A, D).dot(A))
        V = V.T

        freqRespShape = freqResp.shape
        freqResp = freqResp.reshape((-1, freqResp.shape[2]))
        freqResp = np.dot(V.T, freqResp.T).T
        freqResp = freqResp.reshape(freqRespShape)
        freqRespDecorr = freqResp.copy()

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis, np.newaxis, :]
    LPQdesc = ((freqResp > 0) * (2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode == "im":
        LPQdesc = np.uint8(LPQdesc)

    ## Histogram if needed
    if mode == "nh" or mode == "h":
        LPQdesc = np.histogram(LPQdesc.flatten(), range(257))[0]

    ## Normalize histogram if needed
    if mode == "nh":
        LPQdesc = LPQdesc / LPQdesc.sum()

    return LPQdesc

def extract_lpq_features_from_regions(
    image_name: str,
    regions_matrix: np.ndarray, 
    winSize=7, 
    decorr=1, 
    mode="nh"
) -> np.ndarray:
    """
    Extract LPQ features from all regions in a segmented image matrix with caching support.
    
    Args:
        image_name (str): Name of the image file (without extension)
        regions_matrix (np.ndarray): Matrix containing image segments (rows x cols x region_data)
        winSize (int): Window size for LPQ computation (default: 7)
        decorr (int): Decorrelation flag (default: 1)
        mode (str): LPQ mode - "nh" for normalized histogram (default: "nh")
        
    Returns:
        np.ndarray: Array containing LPQ features for all valid regions
    """
        
    # Define cache directory and file path
    cache_dir = Path("features/lpqs")
    cache_file = cache_dir / f"{image_name}_segmented_lpq.npy"
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            features_array = np.load(cache_file)
            return features_array
        except Exception as e:
            print(f"Warning: Failed to load cached segmented LPQ for {image_name}: {e}")
            # Continue to compute LPQ if loading fails
    
    # Compute LPQ for all segments
    features_list = []
    rows, cols = regions_matrix.shape
    
    for row in range(rows):
        for col in range(cols):
            region = regions_matrix[row, col]
            if _is_valid_region(region):
                try:
                    lpq_features = lpq(region, winSize, decorr, mode)
                    features_list.append(lpq_features)
                except Exception as e:
                    print(f"Warning: Failed to compute LPQ for {image_name} region ({row},{col}): {e}")
                    # Add zero array as placeholder for failed regions (expecting 256 features)
                    expected_features = 256  # LPQ typically returns 256-element histogram
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
                    padded_features[:len(features)] = features
                    normalized_features.append(padded_features)
                else:
                    normalized_features.append(features)
            
            features_array = np.array(normalized_features)
        else:
            features_array = np.array(features_list)
    else:
        # Fallback for completely empty image
        expected_features = 256
        features_array = np.array([np.zeros(expected_features)])
    
    # Save to cache
    try:
        np.save(cache_file, features_array)
    except Exception as e:
        print(f"Warning: Failed to save segmented LPQ cache for {image_name}: {e}")
    
    return features_array


def _is_valid_region(region) -> bool:
    """
    Check if a region is valid for LPQ computation.
    
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

def extract_lpq_features_from_segmented_images(
    images: Dict[str, np.ndarray], winSize=7, decorr=1, mode="nh"
) -> Dict[str, np.ndarray]:
    """
    Extract LPQ features from multiple segmented images with caching support.
    
    Args:
        images (Dict[str, np.ndarray]): Dictionary of {image_name: regions_matrix}
        winSize (int): Window size for LPQ computation (default: 7)
        decorr (int): Decorrelation flag (default: 1)
        mode (str): LPQ mode - "nh" for normalized histogram (default: "nh")
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of {image_name: features_array}
    """
    return {
        image_name: extract_lpq_features_from_regions(image_name, regions_matrix, winSize, decorr, mode)
        for image_name, regions_matrix in images.items()
    }

def extract_lpq_features_for_each_category_segmented(
    categories: List[str], 
    images: Dict[str, Dict[str, np.ndarray]], 
    winSize=7, 
    decorr=1, 
    mode="nh"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract LPQ features for segmented images organized by category.
    
    Args:
        categories (List[str]): List of category names
        images (Dict[str, Dict[str, np.ndarray]]): Nested dictionary structure 
                                                  {category: {image_name: regions_matrix}}
        winSize (int): Window size for LPQ computation (default: 7)
        decorr (int): Decorrelation flag (default: 1)
        mode (str): LPQ mode - "nh" for normalized histogram (default: "nh")
        
    Returns:
        Dict[str, Dict[str, np.ndarray]]: Dictionary structure 
                                        {category: {image_name: features_array}}
    """
    return {
        category: extract_lpq_features_from_segmented_images(
            images[category], winSize, decorr, mode
        )
        for category in categories
    }


def extract_lpq_features_for_many_segmented_images(
    images: Dict[str, np.ndarray],
    winSize=7,
    decorr=1,
    mode="nh",
) -> Dict[str, np.ndarray]:
    """
    Extract LPQ features for all segmented images in a single category.
    
    Args:
        images: Dict of {image_name: regions_matrix}
        winSize: Window size for LPQ computation
        decorr: Decorrelation flag
        mode: LPQ mode
        
    Returns:
        Dict structure {image_name: features_array}
    """
    lpqs = {}
    
    for image_name, regions_matrix in images.items():
        lpqs[image_name] = extract_lpq_features_from_regions(
            image_name, regions_matrix, winSize, decorr, mode
        )
    
    return lpqs


def extract_lpq_features_for_segments_by_categories(
    categories: List[str],
    segmented_images: Dict[str, Dict[str, np.ndarray]],
    winSize=7,
    decorr=1,
    mode="nh",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract LPQ features for segmented images organized by category.
    
    Args:
        categories: List of category names
        segmented_images: Dict structure {category: {image_name: regions_matrix}}
        winSize: Window size for LPQ computation
        decorr: Decorrelation flag
        mode: LPQ mode
        
    Returns:
        Dict structure {category: {image_name: features_array}}
    """
    return {
        category: extract_lpq_features_for_many_segmented_images(
            segmented_images[category], winSize, decorr, mode
        )
        for category in categories
    }


# -------------------- Full-image LPQ extraction + caching --------------------
def extract_lpq_features_from_image(
    image_name: str, image_value: np.ndarray, winSize=7, decorr=1, mode="nh"
) -> np.ndarray:
    """
    Extract LPQ features for a single full image with caching.

    Cache file: features/lpqs/<image_name>_lpq.npy
    Returns the LPQ descriptor (histogram or image depending on 'mode').
    """
    cache_dir = Path("features/lpqs")
    cache_file = cache_dir / f"{image_name}_lpq.npy"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        try:
            features = np.load(cache_file)
            return features
        except Exception as e:
            print(f"Warning: Failed to load cached LPQ for {image_name}: {e}")

    features = lpq(image_value, winSize, decorr, mode)

    try:
        np.save(cache_file, features)
    except Exception as e:
        print(f"Warning: Failed to save LPQ cache for {image_name}: {e}")

    return features


def extract_lpq_features_from_images(
    images: Dict[str, np.ndarray], winSize=7, decorr=1, mode="nh"
) -> Dict[str, np.ndarray]:
    """
    Extract LPQ features from multiple full images (not segmented).

    Args:
        images: Dict of {image_name: image_array}
    Returns:
        Dict of {image_name: features_array}
    """
    lpqs = {}

    for image_name, img in images.items():
        lpqs[image_name] = extract_lpq_features_from_image(
            image_name, img, winSize, decorr, mode
        )

    return lpqs


def extract_lpq_features_for_each_category(
    categories: List[str],
    images: Dict[str, Dict[str, np.ndarray]],
    winSize=7,
    decorr=1,
    mode="nh",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract LPQ features for full images organized by category.

    Mirrors the segmented flow but operates on full images and uses caching.
    """
    return {
        category: extract_lpq_features_from_images(
            images[category], winSize, decorr, mode
        )
        for category in categories
    }
