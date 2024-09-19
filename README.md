# LifeThread: Suicide Risk Assessment Using Social Media Posts

## Introduction

LifeThread is a project dedicated to assessing suicide risk based on textual data extracted from social media posts. The goal of this project is to leverage advanced machine learning techniques to identify posts that may indicate high risk of suicide. By analyzing posts from platforms such as Reddit and Suicide Watch, we aim to develop models that can effectively classify the level of risk and provide valuable insights for preventive measures.

The project employs a combination of sentiment analysis and classification models to evaluate the content of the posts. Initially, sentiment analysis was performed to gauge the overall mood of the posts. However, to enhance the precision of risk assessment, the focus shifted towards training classification models that use a variety of features to better predict suicide risk.

Explore this file to learn more about how data was processed, models were trained, and the results achieved in the context of suicide risk assessment.

# Files

This section contains description of files supplied within project.

## Datasets

The project utilizes two datasets related to social media posts concerning depression and suicide.

### Dataset: Reddit Depression and SuicideWatch

**Source:** Kaggle - _Reddit Dataset: Depression and SuicideWatch_

**Description:** This dataset contains posts from Reddit classified as related to depression or suicidal thoughts. These data are used to assess suicide risk based on post content.

### Dataset: Suicide Watch

**Source:** Kaggle - _Suicide Watch Dataset_

**Description:** This dataset includes posts from the Suicide Watch forum, classified for suicide risk. Posts are used to train models evaluating risk based on text.

## Loading and Merging 

After downloading, both datasets were loaded and merged. Label classification was standardized using label mapping, where:

*  _depression_ and _non-suicide_ are labeled as **0 (low risk of suicide)**
*  _SuicideWatch_ and _suicide_ are labeled as **1 (high risk of suicide)**

## Text Processing

1.  **Normalization:** The text was converted to lowercase.
2.  **Duplicate Removal:** Duplicate posts were removed.
3.  **Sentiment Analysis:** Sentiment for each post was assessed using the DistilBERT model, and results were normalized to a uniform distribution using QuantileTransformer.
4.  **Preparations for Training:** Abandonment of sentiment analysis and preparations for classification models training.
5.  **Splitting into Training and Test Sets:** Data was divided into training and test sets for further model training and evaluation.

## Files and Folders

**combined_data.csv:** Combined dataset with labels and processed text.

**combined_data_with_risk.csv:** Dataset with additional sentiment analysis results.
Models/: Folder containing saved models

# Sentiment Analysis

## Description

Sentiment analysis is a crucial step in assessing the risk of suicide based on post content. In this project, sentiment analysis was performed using a transformer-based language model, specifically distilbert-base-uncased-finetuned-sst-2-english. This model was trained on the SST-2 (Stanford Sentiment Treebank) dataset, which is widely used for sentiment analysis tasks.

## Process

Model and Tokenizer Retrieval: Initially, the pre-trained model and tokenizer were loaded from the transformers library. The distilbert-base-uncased-finetuned-sst-2-english model was chosen for its ability to evaluate sentiment on a scale from 0 to 1, corresponding to a range from negative to positive sentiment.

**Text Tokenization:** Each post was tokenized using the BERT tokenizer, meaning the text was divided into tokens that the model could understand. Tokenization ensures that all texts have a uniform length and format, which is critical for the model's proper functioning.

**Sentiment Analysis:** The model analyzed the texts and assigned sentiment values on a scale from 0 to 1. These values represent the degree to which the text is positive. The results were then processed to derive overall conclusions about trends in the data.

**Normalization:** To make sentiment values more consistent and better suited for further analysis, a QuantileTransformer was applied, transforming the sentiment values to a uniform distribution scale.

## Analysis Results

### Sentiment Distribution Histogram

Visualization of the sentiment value distribution revealed that posts classified as suicidal tend to have lower sentiment values compared to non-suicidal posts. Histograms provided a visual representation of these differences.

### Conclusions

Non-suicidal posts (blue) have a higher concentration in the lower sentiment range (0-0.5), implying safer sentiment.
Suicidal posts (orange) cluster around the higher sentiment values (0.5-1), indicating more risky or harmful sentiment.

Sentiment analysis revealed that while the model effectively identifies certain trends and differences in sentiment between posts, sentiment alone is not always sufficient for precise suicide risk classification. Therefore, sentiment analysis was used as one component in a broader risk assessment context rather than as the sole classification tool.

## Role in the Project

Sentiment analysis provides valuable insights into the mood expressed in posts, which can help identify potentially risky content. However, due to the limitations of sentiment analysis alone, these results are used in conjunction with other classification methods to ensure a more accurate risk assessment.


# Transition from Sentiment Analysis to Classification Models

After performing sentiment analysis, it was determined that while sentiment values provided useful insights into the mood of the posts, they were not sufficient on their own for accurate suicide risk classification. Consequently, the focus shifted to developing and training classification models to better assess the risk based on the processed data.

## Rationale for Transition

- **Limitations of Sentiment Analysis**: Sentiment analysis revealed that while it could identify trends and general sentiment levels, it lacked the precision needed for reliable risk classification. Sentiment values alone did not fully capture the complexities of the posts related to suicide risk.

- **Need for More Robust Models**: To address the limitations, the decision was made to use more sophisticated classification models. These models could leverage additional features and patterns in the data beyond sentiment scores to improve accuracy and reliability.

## Outcome

By moving from sentiment analysis to specialized classification models, the project aimed to achieve a more nuanced understanding of the data and improve the accuracy of suicide risk assessments. The new models incorporated multiple features and advanced techniques, leading to better performance in distinguishing between high and low-risk posts.

# Life Thread Classifiers

## Process of Model Training

**Data Preparation**: The data used for sentiment analysis was further processed to extract relevant features for classification. This included combining the sentiment scores with other textual features from the posts.

**Model Selection**: Various classification models were selected and trained, including:
- **Simple ML Algotythms:**
  - **Stochastic Gradient Descent (SGD)**
  - **Naive Bayes**
  - **Support Vector Classification (SVC)**
- **Pre-trained Language Models:**
  - **BERT**
  - **DistilBERT**

**Training and Evaluation**: Each model was trained on the processed dataset and evaluated based on its performance in classifying posts into different risk categories. The performance metrics helped in selecting the most effective model for the task.
