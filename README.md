[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122316&assignment_repo_type=AssignmentRepo)

# XNAP-Project Business Classification

This project aims to classify businesses in street view images using a combination of visual and textual features. The model will be based on the Transformer architecture, which will leverage convolutional neural networks (CNN) and word embeddings to extract information from the images and textual data, respectively.

The objective is to perform fine-grained classification, which involves identifying the specific type of business in the image among 28 different categories. The dataset used for this project consists of 25,255 images from ImageNet and can be found at [https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html](https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html). The starting point for the project is the ConTextTransformer repository on GitHub, which will be customized to suit the specific requirements of this project.

## Objective

Classify businesses on street view images, combining textual and visual features

## Type of Model

Transformer (CNN and Word Embeddings)

## Task

Fine-grained Classification

## Data

25,255 images from ImageNet of 28 categories (do not reuse the old OCR)  -  https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html

## Starting point

https://github.com/lluisgomez/ConTextTransformer

## Code structure

The codebase for this project is structured as follows:

├── data/ 
├── models/ 
│   └── saved_models/ 
├── results/ 
│   ├── train/ 
│   └── test/ 
├── src/ 
│   ├── data/ 
│   ├── models/ 
│   ├── utils/ 
│   ├── visualization/ 
│   ├── train.py 
│   └── test.py 
├── config.py 
├── requirements.txt 
├── README.md 
└── main.py 

The `data` directory will contain all the data files. The `models` directory contains a subdirectory for saved models. The `results` directory contains subdirectories for training and evaluation results. The `src` directory contains the source code for the project, organized into subdirectories for data processing, model architectures, utility functions, visualization, training, and evaluation. The `config.py` file contains configuration parameters for the project, such as hyperparameters for the model and file paths for the data. The `requirements.txt` file lists the dependencies required to run the project. The `README.md` file contains instructions on how to set up and run the project, as well as information on the project's goals and structure. Finally, the `main.py` file is the entry point for running the project.

## How to Run the code?

The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights &amp; Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:

```
conda activate xnap-example
```

To run the example code:

```
python main.py
```

## Bibliografia

* Seref Karaoglu. (2021). Business Image Dataset. Retrieved from [https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html](https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html)
* Gomez, L., & Karacapilidis, N. (2021). ConTextTransformer: Learning with Visual and Textual Contextual Cues. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 2829-2838).

## Contributors

- Abel Espin Romero 1605961@uab.cat
- Biel Castellarnau Ruiz 1599417@uab.cat
- Sergi Garriga Mas 1606989@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau d'Enginyeria de Dades,
UAB, 2023
