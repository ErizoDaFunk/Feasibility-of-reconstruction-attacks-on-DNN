# Ginver: Inversion Attacks on Deep Neural Networks

## Overview

This project explores inversion attacks on deep neural networks, focusing on reconstructing input images from intermediate representations of models such as ResNet-50. The repository includes code for training, attacking, evaluating, and visualizing inversion models, with experiments on both simple datasets (MNIST) and complex datasets (ImageNet-like, EMBL).

# Project Structure

```bash
Feasibility-of-reconstruction-attacks-on-DNN
│
├── Ginver-main-resnet50/
│   ├── attack.py
│   ├── result_attack.py
│   ├── generate_metrics.py
│   ├── Model.py
│   ├── Dataset/
│   ├── ImagResult/
│   ├── ModelResult/
│   └── ...
│
├── Ginver-main-nist/
│   ├── data/
│   ├── Ginver-main/
│   ├── grid_search_results/
│   ├── ImagResult/
│   └── ModelResult/
│
├── dockerfile/
│   ├── Dockerfile
│   └── requirements.txt
│
├── cloudLab/
│   └── ...
│
├── papers/
│   └── ...
│
├── requirements.txt
└── ...
```


## Main Components

- **Ginver-main-resnet50/**: Main codebase for ResNet-50 inversion attacks.
    - `attack.py`: Script for training and running inversion attacks.
    - `result_attack.py`: Visualization of reconstructed images for each layer.
    - `generate_metrics.py`: Computes metrics (MSE, SSIM, accuracy) for reconstructed images.
    - `Model.py`: Model definitions for classifier and inversion networks.
    - `Dataset/`, `ImagResult/`, `ModelResult/`: Data, results, and model checkpoints.
- **Ginver-main-nist/**: Experiments and code for MNIST dataset.
- **dockerfile/**: Docker support for reproducible environments.
- **cloudLab/**: Documentation and inventory for hardware experiments.
- **papers/**: Reference papers and documentation.

## Setup

### 1. Environment

It is recommended to use [Miniconda/Anaconda](https://docs.conda.io/en/latest/) for managing the Python environment.

Install dependencies:
```bash
pip install -r [requirements.txt](http://_vscodecontentref_/3)
```

For Netron export and visualization:
```bash
pip install git+https://github.com/raphael-prevost/netron-export
playwright install --with-deps chromium
```

### 2. Data Preparation

- Place your datasets in the appropriate folders, e.g., `Ginver-main-resnet50/Ginver-main/Dataset/` for ResNet-50 experiments.
- For MNIST, use the structure in `Ginver-main-nist/data/`.

### 3. Training and Attacks

- **Train inversion models**:
    ```bash
    python Ginver-main-resnet50/Ginver-main/attack.py --layer <layer_name> --mode <mode> --cuda --save-model
    ```
    - `--layer`: Layer to attack (e.g., conv1, maxpool, layer1_0, etc.)
    - `--mode`: Attack mode (`whitebox` or `blackbox`)

- **Visualize results**:
    ```bash
    python Ginver-main-resnet50/Ginver-main/result_attack.py
    ```

- **Generate metrics**:
    ```bash
    python Ginver-main-resnet50/Ginver-main/generate_metrics.py
    ```

### 4. Results

- Reconstructed images are saved in `ImagResult/` or `results_visualization/`.
- Metrics (MSE, SSIM, accuracy) are saved in `metrics.csv`.

## Key Features

- **Layer-wise inversion**: Attack and reconstruct images from any intermediate layer.
- **Whitebox & Blackbox modes**: Evaluate attacks with and without knowledge of the model.
- **Comprehensive metrics**: MSE, SSIM, and classification accuracy on reconstructed images.
- **Visualization**: Side-by-side comparison of original and reconstructed images.

## Reproducibility

- All code is provided with configuration files and scripts for reproducibility.
- Docker support is available in the  directory.

## License

- **Code, Documentation & Media**: Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/80x15.png)](https://creativecommons.org/licenses/by/4.0/)

## Acknowledgements

- Based on research in neural network inversion and privacy attacks.
- See  for related literature and references.

---

For questions or contributions, please open an issue or pull request.