# GoEmotions: Emotion Classification using Machine Learning

This repository contains an end-to-end machine learning project to classify text into six main emotion categories (joy, sadness, anger, fear, surprise, and neutral) using the GoEmotions dataset from Google. The project demonstrates key steps of the ML pipeline, including data preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation.

## Project Overview

- **Objective**: Build a text classification model to predict emotions from text.
- **Dataset**: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions), containing over 58,000 labeled text samples.
- **Model**: Logistic Regression with TF-IDF vectorization.
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Steps in the Project

1. **Data Preprocessing**:
   - Cleaned text data by expanding contractions, removing single-character words, and normalizing informal phrases.
   - Mapped 28 original emotion categories into six main emotions.

2. **Feature Extraction**:
   - Used TF-IDF vectorization (unigrams and bigrams) with a maximum of 10,000 features to represent text numerically.

3. **Model Selection**:
   - Chose Logistic Regression for its efficiency and effectiveness in multi-class classification.
   - Handled class imbalance using `class_weight='balanced'`.

4. **Hyperparameter Tuning**:
   - Performed RandomizedSearchCV to optimize hyperparameters such as `C`, `solver`, and `max_iter`.

5. **Evaluation**:
   - Assessed the model using a classification report, confusion matrix, and precision-recall metrics.

## Results

- **Accuracy**: ~57%
- **Insights**:
  - High performance on "neutral" and "joy" classes.
  - Challenges in predicting "fear" and "surprise" due to limited data.

## Repository Structure

- **data/**: Contains preprocessed dataset files.
- **notebooks/**: Jupyter notebooks demonstrating the analysis and model development.
- **scripts/**: Python scripts for data preprocessing, training, and evaluation.
- **outputs/**: Includes results such as confusion matrix and classification report.

## Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GoEmotions-ML-Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd GoEmotions-ML-Project
   ```
3. Run the preprocessing and training scripts:
   ```bash
   python scripts/preprocess.py
   python scripts/train_model.py
   ```

## References

- Dataset: [Google GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- Libraries: Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn
- Research Papers:
  - Google Research's [GoEmotions](https://arxiv.org/abs/2005.00547)

Feel free to contribute or raise issues in the repository!
```

You can copy and paste this file as `README.md` in your GitHub repository. Make sure to update placeholders like `yourusername` with your actual GitHub username.
