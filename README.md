# Fake News Detection

## Overview
This project aims to classify news articles as "Fake" or "True" using machine learning techniques. The dataset consists of two CSV files: `Fake.csv` and `True.csv`, which contain labeled fake and true news articles, respectively. The classification models include Random Forest, Gradient Boosting, and Logistic Regression.

## Features
- Data preprocessing (cleaning, stemming, and stopword removal)
- Data visualization using Seaborn and Matplotlib
- Model training and evaluation using Scikit-learn
- Confusion matrix and classification reports
- Cross-validation for model performance assessment

## Dependencies
To run this project, install the required dependencies:
```bash
pip install pandas numpy nltk seaborn matplotlib scikit-learn wordcloud
```

## Dataset
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains true news articles
- `df_manual`: A manually curated subset of data for testing

## Data Preprocessing
- Removed duplicate entries
- Filled missing values
- Removed `(Reuters)` mentions
- Applied stemming and stopword removal
- Transformed text using TF-IDF vectorization

## Model Training & Evaluation
The following models were trained and evaluated:

### 1. **Random Forest Classifier**
- Trained on TF-IDF features
- Accuracy: ~99% (train), ~97% (test)

### 2. **Gradient Boosting Classifier**
- Accuracy: ~96% (test)

### 3. **Logistic Regression**
- Accuracy: ~95% (test)

## Visualizations

## Manual Testing
A function `manual_testing(news)` allows users to input custom news articles for classification.
Example usage:
```python
news_article = """Sample news text here..."""
print(manual_testing(news_article))
```

## Cross-Validation
The model was validated using Stratified K-Fold Cross Validation:
```python
Cross Validation Scores: [0.97, 0.96, 0.98, 0.97, 0.97]
Average Score: 0.97
Best Score: 0.98
Worst Score: 0.96
```

## Saving the Model
The final model was saved using a pipeline:
```python
import pickle
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(random_forest_pipeline, f)
```

## Future Work
- Implement a web interface for real-time news classification
- Experiment with deep learning models such as LSTMs and Transformers
- Improve data preprocessing techniques for better accuracy

## Author
Muhammad Asfand Yar
