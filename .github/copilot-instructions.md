# TCC Image Classification by Relevance of Regions - AI Agent Guide

## Project Overview
This is a computer vision thesis project implementing image classification using region relevance analysis. The system segments images into grids and applies texture analysis (LBP, GLCM, LPQ) to identify which regions are most relevant for classification.

## Architecture & Key Components

### Core Workflow (main.ipynb)
The main Jupyter notebook orchestrates the complete pipeline:
1. Load and preprocess images (grayscale conversion, resizing with padding)
2. Dynamic image segmentation using configurable K values (MIN_K=4, MAX_K=20, BASE_SIZE=512)
3. Feature extraction using three texture descriptors in parallel
4. Cross-validation with specialist models for each category
5. Relevance analysis to identify important image regions

### Feature Extraction Modules
- **lbp.py**: Local Binary Pattern with caching (`features/lbps/`)
- **glcm.py**: Gray-Level Co-occurrence Matrix with joblib parallelization
- **lpq.py**: Local Phase Quantization using STFT-based texture analysis

### Tools Package Architecture
- **tools/image_tools.py**: Core image processing (segmentation, resizing, loading)
- **tools/relevance.py**: Model training and relevance score extraction using cross-validation
- **tools/specialists.py**: Binary classifier construction (class vs. non-class)
- **tools/data.py**: Dataset preparation, fold generation, feature combination

### Data Organization
```
images/train/     # Original training images by category
images/test/      # Test images
features/glcms/   # Cached GLCM features per image
features/lbps/    # Cached LBP histograms
features/lpqs/    # Cached LPQ descriptors
```

## Development Patterns

### Environment Setup
Use `uv` for dependency management (preferred) or traditional pip with venv. The project uses Python 3.13+ with scientific computing stack (scikit-image, sklearn, matplotlib, numpy).

### Feature Caching Strategy
All feature extractors implement disk caching in `features/` subdirectories using numpy `.npy` files. Always check cache before recomputing expensive operations.

### Type System (mytypes.py)
The project uses comprehensive TypedDict definitions:
- `ClassificationDataset`: List of cross-validation folds
- `SpecialistSet`: Binary classification data structure
- `ModelResults`: Region-wise probability mappings

### Configuration via Environment Variables
Key parameters are configurable via environment variables:
- `IMAGE_MIN_K`, `IMAGE_MAX_K`: Segmentation bounds
- `IMAGE_BASE_SIZE`, `IMAGE_BASE_K`: Dynamic K calculation parameters

### Cross-Validation Pattern
Use `tools/data.py` functions for consistent fold generation. The system implements specialist models (one-vs-all) with proper train/test separation and probability extraction.

## Critical Implementation Details

### Image Segmentation
- Uses `segment_image_into_grid()` for uniform grid division
- Dynamic K calculation based on image dimensions and base parameters
- Supports both automatic and manual segmentation factor selection

### Parallel Processing
- GLCM extraction uses joblib for parallel computation
- Feature extraction is cached to avoid recomputation
- Memory-efficient handling of large feature matrices

### Model Training Workflow
1. Build specialist sets using `build_specialist_set()`
2. Generate cross-validation folds with `build_classification_dataset()`
3. Extract model probabilities using `extract_model_results()`
4. Analyze region relevance from probability distributions

### Debugging & Visualization
- Extensive matplotlib visualizations for feature distributions
- Image segmentation preview with `visualize_image_segmentation_auto()`
- Warning suppression for joblib ResourceTracker noise

## Important Conventions

- Always reload modules with `importlib.reload()` in notebook cells
- Use absolute paths for file operations
- Feature arrays are numpy arrays with consistent dimensionality
- Category names follow pattern: cat1-20, dog1-20, horse* (various IDs)
- Image preprocessing includes grayscale conversion and padding to maintain aspect ratio