# ACM SIGSPATIAL Cup 2023 - DeepLabV3Plus

## Introduction

This is a submission for the [GISCUP 2023 competition](https://sigspatial2023.sigspatial.org/giscup/),
seeking to automate the detection of surface lakes on the Greenland ice sheet from satellite images in order
to allow scientists to easily track the behavior of lakes through repeated summer melt seasons.

### Authors

Simon Walther, Leonard Cseres, Rémy Marquis, Bertil Chapuis, Andres Perez-Uribe

### Who we are

We are a small team of engineers and data scientists. We are participating in the GISCUP competition under
the umbrella of the [HEIG-VD](https://heig-vd.ch/) and [HES-SO](https://www.hes-so.ch/) and in collaboration with
the [Swiss AI Center](https://www.hes-so.ch/swiss-ai-center).

HEIG-VD, or Haute École d'Ingénierie et de Gestion du Canton de Vaud, is a renowned institution of
higher education in Switzerland, specializing in engineering and management disciplines. It is an
integral part of the HES-SO, the University of Applied Sciences and Arts Western Switzerland,
offering a diverse range of practical and industry-focused programs.

The mission of the Swiss AI Center is to accelerate the adoption of artificial intelligence in the
digital transition of Swiss SMEs.

### Hardware Used for Model Training

- **CPU:** Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- **CPU Cores utilized**: 34
- **GPUs**: 4 \* NVIDIA A40, approx. 80GB VRAM total
- **RAM:** 125 GB

### Reproducibility

The provided code and present README are designed to facilitate experiment replication. If you need assistance or
encounter any issues, please feel free to contact us. We will respond promptly to assist you.

### Software Environment

- **OS:** Linux 5.4.0-149-generic #166-Ubuntu SMP x86_64 GNU/Linux
- **Distribution:** Ubuntu 20.04.3 LTS
- **Python:** 3.10.12
- **CUDA on Machine:** 11.3

## Installation

After cloning the repository, create a virtual environment and install the dependencies:

```sh
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

Or with conda

```sh
conda create -n giscup2023 python=3.10.12 pip
conda activate giscup2023
```

Then install requirements
```sh
# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contains only the top-level dependencies.
You can check transitives dependencies versions in `requirements_freeze.txt` file. 

Note that code has been tested with python 3.10.12 and CUDA 11.3.
Tensorflow is included in `requirements.txt` but depending on your configuration and CUDA version, you may need to follow this: [Install TensorFlow with pip](https://www.tensorflow.org/install/pip?hl=en)

### Download Data from DVC

You can pull the data from the remote storage with the following command:

```sh
dvc pull
```

## Development Environment

In order to setup the development environment, you need to install the pre-commit hooks:

```sh
# Install pre-commit hooks
pre-commit install
```

### Modify DVC Remote Storage

In order to use a different remote storage (in this case we use Google Cloud), you can to run the following:

```sh
dvc remote modify data url gs://<mybucket>/<path>
```

If your bucket is private, you need to set the path to Google Cloud Service Account key:

```sh
dvc remote modify --local myremote \
  credentialpath 'path/to/credentials.json'
```

Alternatively, you can manually set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the credentials file:

```sh
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

You can see more information about DVC remote storage [here](https://dvc.org/doc/user-guide/data-management/remote-storage/google-cloud-storage).

### Reproducing experiment

```sh
dvc repro
```

> **Note**
> Since DVC caches the results of the pipeline, you can run `dvc repro --force` to
> regenerate the results, even if no changes were found.

## Method

### Dataset creation

We extract overlapping tiles over all the training regions and images while filtering tiles having more than 60% of black pixels. Tiles are extracted at size 448x448 and downsampled to 320x320.

All lakes tiles are kept, but we add in the dataset as many tiles without lakes than with. To select the non-lake tiles, we first run a k-means (k = 4) clustering over their HSV histograms (5 bins on each channels). Then we use clusters sizes, with an inversely-proportional probability, to pick non-lake tiles randomly. By doing so, we hope to have more chance to pick diverse tiles.

In the final version, no validation set what used to have a maximum amount of training data.

### DeepLabV3Plus

Semantic segmentation model architecture is DeepLabv3+ (https://arxiv.org/abs/1802.02611) with some modifications. Implementation has been taken and adapted from https://keras.io/examples/vision/deeplabv3_plus/.

Compared to the aforementioned implementation:
- We reduced the number of filters
- We added spatial dropout (https://arxiv.org/pdf/1411.4280v3.pdf)
- ReLU activation function has been replaced by GELU (https://arxiv.org/abs/1606.08415)
- BatchNormalization has been replaced by GroupNormalization (https://arxiv.org/pdf/1803.08494.pdf)

## Model training

Chosen optimizer is SGD with momentum=0.99, clipnorm=1. Snapshot ensembles method (https://arxiv.org/abs/1704.00109) is used to have an ensemble of three models.

### Pre-trained models

The trained models are available on [huggingface.co/swiss-ai-center](https://huggingface.co/swiss-ai-center/giscup2023-deepLabV3Plus).

## Inference

Tiles are extracted with overlap and predictions are averaged across overlaps and the models ensemble predictions.

## Developer Reference

### Pipeline

The pipeline is defined in the `dvc.yaml` file. It contains the following stages:

- `prepare` - Data preparation
- `preprocess` - Data preprocessing
- `segmentation` - Training of semantic segmentation model
- `process` - Prediction on all regions and images
- `postprocess` - Data post-processing
- `evaluate` - GPKG prediction evaluation (/!\\ not the test evaluation, this evaluation is based on validation **and training data**)

The `prepare` and `preprocess` should not be modified in the `dvc.yaml` file. Instead, your implementation should be added in the `PREPARE_METHODS` and `PREPROCESS_METHODS` dictionaries in `src/prepare.py` and `src/preprocess.py` accordingly as well as updating the parameters in the `params.yaml` file.

The `process` step can be modified in the `dvc.yaml` file and even new stages can be added in between.

The `postprocess` and `evaluate` steps should not be modified in the `dvc.yaml` file as they are conformed to the competition rules.

### Parameters

The `params.yaml` file contains the parameters used in the pipeline. It contains two "sections":

- Global parameters
- DVC pipeline parameters

#### Adding New Method

In order to add a new method, you need to add a new entry in the corresponding dictionary in `src/prepare.py` or `src/preprocess.py` and update the `params.yaml` file accordingly.

For example, update the `src/prepare.py` file:

```python
from methods.prepare.<my_custom_method> import my_custom_method # Import your implementation


PREPARE_METHODS = {
    "default": lambda: None,
    "my_custom_method": my_custom_method, # Add your implementation to the dictionary
    # ...
}
```

Update the `params.yaml` file:

```yaml
prepare:
  method: <my_custom_method>
  file_path: <path_to_file_containing_method_implementation>
  out: <output_path_of_current_method>

  default: # Keep this section empty

  <my_custom_method>: # Define your custom method here
    <my_custom_method_param>: <my_custom_method_param_value> # Define your method parameters here
    # ...
```

> **Note**
> All the parameters defined in your custom method will passed as `kwargs` to the `my_custom_method` Python function you implemented.

## Tools

- [Docker](https://www.docker.com/)
- [git](https://git-scm.com/)

### GeoSpatial

- [shapely](https://shapely.readthedocs.io/en/stable/manual.html)
- [geopandas](https://geopandas.org/)
- [rasterio](https://rasterio.readthedocs.io/en/latest/)

### Machine Learning

- [opencv-python](https://pypi.org/project/opencv-python/)
- [tensorflow](https://www.tensorflow.org/)
- [albumentations](https://albumentations.ai)
- [scikit-learn](https://scikit-learn.org/)

### Data

- [dvc](https://dvc.org/)
- [cml](https://cml.dev/)

## Resources

- [ACM SIGSPATIAL Cup 2023](https://sigspatial2023.sigspatial.org/giscup/index.html)
- [GisCup 2023 - Google Drive](https://drive.google.com/drive/folders/1p5N7QQwNkC5is89_IfdQfOZ__dQia91x)
