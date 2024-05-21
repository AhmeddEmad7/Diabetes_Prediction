# Diabetes Prediction Using Machine Learning

## Overview
This project aims to predict diabetes among individuals using various machine learning models. The dataset used in this project includes multiple health-related features that might influence the diabetes condition. The project explores different aspects of data preprocessing, exploratory data analysis, feature selection, model training, and evaluation to develop a robust predictive model.

## How to Run This Project
1. Clone the repository to your local machine.
2. Ensure you have Jupyter Notebook installed, or use Google Colab to open the `.ipynb` file.
3. Install the required libraries mentioned in the `requirements.txt` file.
4. Run the notebook cells sequentially to reproduce the analysis and results.

## Dependencies
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn

## Dataset
The dataset, `diabetes_data.csv`, includes health-related metrics such as blood pressure, cholesterol levels, body mass index (BMI), and other lifestyle factors. The target variable, `Diabetes_012`, indicates the diabetes status: 0 for non-diabetic, 1 for pre-diabetic, and 2 for diabetic.

## Data Preprocessing
Data preprocessing steps included:
- Handling missing values: The dataset was checked for null values.
- Removing duplicates: Duplicate entries were identified and removed to ensure the quality of the dataset.
- Feature selection: Irrelevant features were dropped based on domain knowledge and correlation analysis.

## Exploratory Data Analysis (EDA)
- **Distribution Analysis**: Visualizations such as histograms and bar plots were used to understand the distribution of various features.
- **Correlation Analysis**: A heatmap was generated to explore the correlations between different features and the target variable.
- **Class Imbalance Handling**: Techniques like undersampling the majority class and oversampling the minority class were employed to handle class imbalances.

## Feature Engineering
- Continuous and categorical features were identified and appropriately processed.
- New features were engineered from existing data to improve model performance.

## Model Building
Several machine learning models were trained and evaluated:
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **AdaBoost Classifier**
- **XGBoost Classifier**
- **Linear Discriminant Analysis (LDA)**

## Model Evaluation
Each model was evaluated using metrics such as the confusion matrix and classification report. The models' performance was primarily assessed based on their recall, precision, and F1-score, particularly focusing on the macro average to account for class imbalances.

## Results
The models showed varying degrees of success, with ensemble methods like **SVM** and **AdaBoost** performing particularly well due to their ability to handle non-linearities and interactions between features effectively.

## Conclusion
The project successfully demonstrates the application of various machine learning techniques to predict diabetes. Future work could explore more sophisticated feature engineering, the use of deep learning models, and deployment of the model as a web-based application for real-time predictions.

## Additional Resources
- A comprehensive paper detailing the methodologies and findings of this project is also available in the repository under the name `Diabetes_Prediction_Paper.pdf`.

## Authors

<div align="left">
    <a href="https://github.com/AhmeddEmad7">
    <img src="https://github.com/AhmeddEmad7.png" width="100px" alt="@AhmeddEmad7">
  </a>
 <a href="https://github.com/nourhan-ahmedd">
      <img src="https://github.com/nourhan-ahmedd.png" width="100px" alt="@nourhan-ahmedd">
    </a>
    <a href="https://github.com/ZiadMeligy">
      <img src="https://github.com/ZiadMeligy.png" width="100px" alt="@ZiadMeligy">
    </a>
    <a href="https://github.com/MayarAhmeddd">
      <img src="https://github.com/MayarAhmeddd.png" width="100px" alt="@MayarAhmeddd">
    </a>
</div>