# Multi-class Text Classification with Transformers

This repository presents a detailed exploration and implementation of Transformer models for the task of multi-class text classification. Leveraging the extensive 20 Newsgroups dataset, this project showcases the end-to-end process of applying advanced NLP techniques to categorize text into one of 20 distinct categories.

## Overview

The project is structured around several core components, each addressing a critical aspect of machine learning workflow:

- **Data Preparation**: Utilizing the 20 Newsgroups dataset, we meticulously preprocess the text data, ensuring it is tokenized, lowercased, and stripped of punctuation to facilitate effective model training. Labels are also transformed to suit the multi-class classification framework.

- **Model Architecture**: At the heart of our solution is a Transformer model, fine-tuned from the pre-trained BERT model `google/bert_uncased_L-4_H-256_A-4`. We have implemented a custom self-attention mechanism within our Transformer model to enhance its understanding of the context and relevance of words in text documents.

- **Training Process**: Our training regimen includes careful monitoring to save the model at its peak performance, ensuring we capture the most effective version of the model based on its validation set performance.

- **Evaluation and Metrics**: The model's prowess is rigorously evaluated on a separate test set, with a comprehensive report on accuracy, precision, recall, and F1 score for each category, alongside an aggregate performance metric.

## Results

The project achieved notable success, demonstrating the Transformer model's capability to effectively classify text across multiple categories. Key performance metrics on the test set include:

- Accuracy: 74.65%
- Precision: 68.98%
- Recall: 69.08%
- F1 Score: 66.49%

These results underscore the model's robustness and its ability to generalize across a diverse set of topics.

## Project Files Overview

- `main.py`: Entry point for training and evaluating the model.
- `models.py`: Defines the Transformer model architecture, including the custom self-attention mechanism, and training/validation steps.
- `dataloader.py`: Contains utilities for loading and preprocessing the dataset.
- `requirements.txt`: Lists all the necessary Python packages.

## Usage

To replicate this project's environment and results, follow these steps:

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python main.py --epochs 7 --batch_size 32`
3. Evaluate the model: Automatically done post-training as part of the `main.py` script.

## Challenges and Solutions

Throughout this project, we encountered and overcame several challenges, including optimizing the custom self-attention mechanism for our specific task and managing computational resources. These hurdles were addressed through iterative testing.

## Conclusion

This project stands as a testament to the power of Transformer models in handling complex NLP tasks. Through careful implementation and fine-tuning, including the development of a custom self-attention mechanism and leveraging a specific pre-trained BERT model, we've demonstrated that even with the vast and varied 20 Newsgroups dataset, it is possible to achieve meaningful and practical text classification results.

## Authors

- Ethan Pirso