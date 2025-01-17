# ProtoCulturalDiversity

ProtoCulturalDiversity is a the code base for the paper "Embedding Cultural Diversity in Prototype-based Recommender Systems." by Armin Moradi, Nicola Neophytou, Florian Carichon and Golnoosh farnadi at ECIR 2025. This repository contains code and resources for dataset visualizations and processing, training models, hyperparameters.

## Table of Contents

- [Citations](#citations)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Citations
```
@article{moradi2024embedding,
  title={Embedding Cultural Diversity in Prototype-based Recommender Systems},
  author={Moradi, Armin and Neophytou, Nicola and Carichon, Florian and Farnadi, Golnoosh},
  journal={arXiv preprint arXiv:2412.14329},
  year={2024}
}
```
## Installation

To get started with ProtoCulturalDiversity, clone the repository and install the required dependencies:

```bash
git clone https://github.com/moradiarmin/ProtoCulturalDiversity.git
cd ProtoCulturalDiversity
conda env create -f ProtoCulturalDiversity.yml
conda activate ProtoCulturalDiversity
```


## Data

To work with the datasets in this project, follow these steps:

1. **Download the Datasets**  
   Instructions for downloading all three datasets are provided in `./data/README.md`.

2. **Initial Data Processing**  
   Use the scripts in the `dataset_processing` folder to process the raw datasets as needed.

3. **Final Dataset Preprocessing**  
   After the initial processing, pre-process the datasets using the respective `splitter.py` files:

   - Navigate to the dataset folder:  
     ```bash
     cd <dataset_folder>
     ```
   - Run the dataset splitter:  
     ```bash
     python <dataset_name>_splitter.py -lh <folder_where_the_data_is>
     ```
     - Typically, the data is already in the folder, so you can use `./` as the path.  
     - Leave the saving path as `./`.

4. **Generated Files**  
   After running the scripts, you will have the following files:
   - Three files containing the listening history of users for training, validation, and testing.
   - Two files containing the user and item IDs (used as indices in the rating matrix).

These processed datasets are ready to be used with the recommendation systems implemented in the repository.


## Usage

After installing the necessary dependencies and finalzing the dataset, you can begin using the project by running the `start.py` script:

```bash
python start.py -m <model> -d <dataset>
```

This script serves as the entry point for the project and will guide you through the available functionalities.

## Project Structure

The repository is organized as follows:

- `confs/`: Configuration files including the optimal hyperparameter sets of the vanilla models.
- `feature_extraction/`: Models.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and experimentation.
- `rec_sys/`: Implementation of a general recommendation systems.
- `utilities/`: Utility functions and helper scripts.
- `experiment_helper.py`: Helper functions for managing experiments.
- `start.py`: Main script to initiate the project.

## Acknowledgements
This codebase extends and builds upon the outstanding work presented in "ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations" by Alessandro B. Melchiorre, Navid Rekabsaz, Christian Ganhör, and Markus Schedl at RecSys 2022.
```
@inproceedings{melchiorre2022protomf,
    title = {ProtoMF: Prototype-based Matrix Factorization for Effective and Explainable Recommendations},
    author = {Melchiorre, Alessandro B. and Rekabsaz, Navid and Ganhör, Christian and Schedl, Markus},
    booktitle = {Sixteenth ACM Conference on Recommender Systems},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    series = {RecSys '22},
    doi = {10.1145/3523227.3546756},
    isbn = {978-1-4503-9278-5/22/09}
}
```
Funding support for project activities has been partially provided by Canada CIFAR AI Chair, and Facebook award. We also express
our gratitude to Compute Canada and Mila for their support in providing compute resources, software and technical help for our evaluations.


## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details. 



