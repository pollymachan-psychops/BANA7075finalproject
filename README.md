# SparkSync Clinical Data Analysis Project

## Project Overview
The SparkSync Clinical Data Analysis project aims to leverage machine learning techniques to analyze clinical data for improved decision-making and insights in patient care. This project specifically focuses on analyzing datasets related to clinical programs and patient coordinator comfort levels.

## Clinical Data Analysis Focus
This project emphasizes two main analysis streams: classification and regression. The classification task addresses the `last_elected_program` to predict the eventual program selection, while the regression task relates to `coordinator_comfort` to quantify the comfort level of coordinators working with patients.

## Dual ML Pipelines
The analysis is conducted through dual machine learning pipelines, with the following details:
1. **Classification Pipeline**:  This pipeline utilizes various classification algorithms to predict the `last_elected_program` using relevant features extracted during data preprocessing.
2. **Regression Pipeline**: For predicting `coordinator_comfort`, this pipeline employs regression techniques suitable for continuous outcome variables.

## Data Cleaning and Preprocessing Steps
- Removal of duplicates and irrelevant features.
- Handling of missing values through appropriate imputation methods.
- Encoding categorical variables into numerical formats.
- Scaling of numerical features for better algorithm performance.
- Splitting the dataset into training and testing sets to evaluate model performance.

## Model Comparison Results
A variety of models were tested for both classification and regression tasks. The performance metrics considered include accuracy, precision, recall for classification, and RMSE, R² for regression, allowing for thorough comparisons of model effectiveness.

## Feature Importance Analysis
The feature importance analysis illustrates which variables contribute most significantly to model predictions, aiding in understanding the underlying factors influencing both the program elections and comfort levels of patient coordinators.

## Installation and Usage Instructions
To successfully run the project, follow the steps below:

1. **Clone the Repository**
   ```
   git clone https://github.com/pollymachan-psychops/BANA7075finalproject.git
   cd BANA7075finalproject
   ```

2. **Install Dependencies**
   Make sure you have Python 3.x installed, then install the necessary packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Scripts**
   - To generate the SparkSync data, run:
     ```
     python bana7075generatesparksyncdata.py
     ```
   - To perform data analysis, run:
     ```
     python sparksync_analysis.py
     ```

## Output Directory Structure
The outputs from the analysis are organized as follows:
```
output/
├── classification_results/
│   ├── model_1_results.csv
│   ├── model_2_results.csv
├── regression_results/
│   ├── regression_analysis_results.csv
└── feature_importance/
    └── feature_importance_plot.png
```

## Technologies Used
- Python for data analysis.
- Scikit-learn for machine learning models.
- Pandas for data manipulation.
- Matplotlib and Seaborn for visualization.
- Spark for handling large datasets efficiently.

## How to Reproduce Results
To reproduce the results, ensure that you have the same environment as specified in `requirements.txt`, and follow the installation and usage instructions above. The dataset must be structured similar to how it was initially prepared for the analysis to ensure compatibility with the scripts.
