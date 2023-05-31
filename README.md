[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122316&assignment_repo_type=AssignmentRepo)

# XNAP-Project Business Classification

This project aims to classify businesses in street view images using a combination of visual and textual features. The model will be based on the Transformer architecture, which will leverage convolutional neural networks (CNN) and word embeddings to extract information from the images and textual data, respectively.

The objective is to perform fine-grained classification, which involves identifying the specific type of business in the image among 28 different categories. The dataset used for this project consists of 25,255 images from ImageNet and can be found at [https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html](https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html). The starting point for the project is the ConTextTransformer repository on GitHub, which will be customized to suit the specific requirements of this project.

## Objectives

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
├── results/
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── conTextDataset.py
│   ├── models/
│   │   ├── conTextTransformer.py
│   ├── utils/
│   │   ├── utils.py
│   │   ├── utils_visualization.py
│   │   └── train_test_labels_split.py
│   ├── config.py
│   ├── hyperparameter_tuning.py
│   ├── inference.py
│   ├── main.py
│   ├── train.py
│   └── test.py
├── environment.yml
├── README.md
└── LICENSE

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

## Firsts executions

During the initial executions of the code, the model was trained on the MNIST dataset using a simple CNN architecture. The purpose of these executions was to set up the environment, verify the code's functionality, and ensure that the dependencies were properly installed.

### Different CNN pretrained models used

- ResNet50 (Residual Network) is a variant of the ResNet (Residual Network) architecture consisting of 50 layers. ResNet was introduced to address the problem of performance decay as more layers are added to a deep neural network. It uses residual blocks that allow direct information flow across connection hops, making it easier to train deeper networks and improving accuracy.
- SE_ResNeXt101 (Residual Network with Next) is an extension to the ResNet of 101 layers that focuses on the idea of "cardinal grouping" to further improve performance. Rather than having a single path of information within each residual block, ResNeXt uses multiple paths, called "transformation paths," which capture different forms of information and allow the residual blocks to be richer and more expressive. These transformation paths are realized by grouping the outputs of the previous block into multiple channels, which is known as "cardinality". Cardinality refers to the number of channels in each grouping. Furthermore, it combines the ideas of ResNet residual blocks with Squeeze-and-Excitation (SE) modules. SE modules were introduced to improve the network's attention span to specific features and highlight relevant information. These modules use a "squeeze" operation to reduce the dimension of the feature maps and an "excitation" operation to model the interdependencies between the channels and recalibrate them based on their importance.
  -ShuffleNet is a convolutional neural network architecture designed for efficient computation and parameter reduction. It achieves efficiency by utilizing group convolutions and channel shuffling operations. The main idea behind ShuffleNet is to divide the input channels into groups and perform separate convolutions on each group. This reduces the number of parameters and computation compared to standard convolutions. After the group convolutions, a channel shuffling operation is applied to mix the information across different groups, allowing the network to capture diverse features. The channel shuffling operation involves rearranging the feature maps by grouping the channels from different groups together. This helps in enhancing the representation power of the network while maintaining a low computational cost. By using group convolutions and channel shuffling, ShuffleNet achieves a good balance between accuracy and efficiency. It has been shown to perform well on various tasks, especially in scenarios with limited computational resources such as mobile devices and embedded systems.

### Different Word-Embedding pretrained models used

- FastText is a model developed by Facebook AI Research that is based on vector representations of words (word embeddings). Unlike other embedding models, such as Word2Vec, FastText represents each word as a sum of its subword embeddings. This allows capturing morphological information and the relationship between compound words and their components. In addition, FastText includes a linear classification layer that allows you to perform text classification tasks. Their characteristics are: Model subwords and morphological relationships. It is efficient in terms of memory and training speed. Suitable for the classification of short texts and languages with limited resources. It does not capture the order of words in a sentence.
- BERT (Bidirectional Encoder Representations from Transformers) is a model developed by Google that is based on the Transformer architecture, which allows capturing long-range relationships in a text. Unlike traditional NLP approaches that process text sequentially, BERT uses a pretrained language model that learns bidirectional representations of words. This means that BERT considers the context both to the left and to the right of a word in a sentence. Their characteristics are: Models the bidirectional context of words in a sentence. Captures long-range relationships and syntactic dependencies. It is capable of performing natural language processing tasks without the need for additional fine-tuning. It requires more computational resources and training time than FastText.

### Different ways to calculate the loss

- Log Softmax + Negative Log Likelihood Loss: Is the default way to calculate the loss provided in our base project. The activation function F.log_softmax is applied to obtain the model outputs in the form of log-probabilities. Then, the F.nll_loss function (Negative Log Likelihood Loss) is used to calculate the loss based on these log-probabilities and the true target class.
- Cross-Entropy Loss: The CrossEntropyLoss function is directly used, which internally combines the Softmax activation function and the cross-entropy loss function into a single operation. This function calculates both the Softmax activation and the cross-entropy loss, eliminating the need to apply F.log_softmax separately.
- The hypothesis behind the change of way to calculate the loss is as follows: CrossEntropyLoss because it offers several advantages over the previous approach. Firstly, it provides greater efficiency by eliminating the need to explicitly apply the F.log_softmax function. This saves computational time and resources since CrossEntropyLoss internally combines the Softmax activation and loss calculation into a single operation. Additionally, CrossEntropyLoss enhances numerical stability compared to the previous sequence of F.log_softmax and F.nll_loss. It handles potential issues of overflow or underflow more effectively, ensuring more reliable and accurate loss calculations. Furthermore, using CrossEntropyLoss improves code clarity and simplicity. It allows for a cleaner implementation by directly utilizing a specific loss function designed for multi-class classification tasks. This approach conveys the intention of the loss calculation more clearly, making the code easier to read and understand.

### Different Learning Rate Schedulers used

- MultiStepLR: The MultiStepLR scheduler is the default scheduler provided in our base project. It is a learning rate adjustment strategy that reduces the learning rate at specific moments during training. It requires a list of milestones (epochs) and reduces the learning rate by a factor of gamma at each of these milestones.
  In our case, we set a list of milestones [15, 30] and a gamma factor of 0.1. This means that the learning rate was reduced by a factor of 0.1 at epochs 15 and 30.
- ReduceLROnPlateau: The ReduceLROnPlateau scheduler is a learning rate adjustment strategy that reduces the learning rate when a model's improvement reaches a plateau. It monitors a metric, such as validation loss, and if no improvement is observed after a certain number of epochs, it reduces the learning rate by a predetermined factor. The ReduceLROnPlateau scheduler takes into account the evolution of the metric of interest and dynamically adjusts the learning rate based on that information. Therefore, if the model has reached a plateau and is not improving, the scheduler will reduce the learning rate to allow for more precise adjustments and potentially escape of local min.
  In our case, the ReduceLROnPlateau scheduler will dynamically adjust the learning rate based on the loss performance. We have set a threshold of 0.1 to indicate that the loss needs to improve by at least 1% to be considered an improvement. If no improvement is observed in the loss for 3 consecutive epochs (patience), the learning rate will be reduced by a factor of 0.1.
- The hypothesis behind the change in scheduler is as follows:
  The MultiStepLR scheduler is a simple but effective strategy to reduce the learning rate at predefined moments during training. However, it has a significant drawback: it does not consider whether the model has reached a plateau or is not improving. This means that the reduction in the learning rate occurs fixedly, regardless of the actual training situation.
  On the other hand, the ReduceLROnPlateau scheduler offers greater flexibility and adaptability. It closely monitors the metric of interest and reduces the learning rate when a stagnation in the model's performance is detected. This allows for more precise and timely adjustments to the learning rate, which can help avoid local minima and achieve more efficient convergence.

## Tests done and Observations

Several tests were conducted to validate the code and make observations. These tests included running the training process and evaluating the performance of the model on the MNIST dataset. The observations made during these tests are:

- The Weights & Biases package is integrated into the code, allowing for monitoring and tracking of the network's learning progress.
- The code structure follows a modular organization, separating different functionalities into respective directories.
- The code utilizes pretrained CNN and word embedding models such as ResNet50 and FastText.
- The ConTextTransformer architecture is used to combine visual and textual features for fine-grained classification.

## To Improve

Based on the initial executions and observations, some potential areas for improvement are identified:

- Explore different CNN pretrained models such as SE_ResNeXt101 to compare their performance.
- Experiment with different word-embedding pretrained models such as BERT to capture more contextual information.
- Fine-tune the hyperparameters of the model to optimize its performance on the specific business classification task.
- Implement data augmentation techniques to increase the diversity and size of the training dataset.
- Consider applying techniques like transfer learning or domain adaptation to improve the model's generalization to street view images.

## Weight & Bias

The Weights & Biases package is integrated into the code to monitor the network's learning progress. It provides visualizations and metrics to track the model's performance during training. By utilizing Weights & Biases, it becomes easier to analyze and interpret the training process, identify potential issues, and make informed decisions for model improvement.


PHOTOS OF THE METRICS AND RESULTS


## Results


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
