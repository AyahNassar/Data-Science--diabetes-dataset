# diabetes_project
## Introduction
This project focuses on the design and implementation of a machine learning model to predict the onset of diabetes within five years in Pima Indians given medical details. This is a binary classification problem, where the outcome '1' denotes the patient having diabetes, and '0' denotes the patient not having diabetes.

## Dataset
The data used in this project is originally from In this project, I analyzed the Pima-Indians-Diabetes-Data using Python’s Pandas, numpy, matplotlib, seaborn, scipy, scikit learn. The dataset can be downloaded from kaggle (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download). This dataset consists of several medical predictor (independent) variables and one target (dependent) variable, Outcome. 

The columns of this dataset are as follows:

1. Pregnancies — Number of times pregnant
2. GlucosePlasma — glucose concentration 2 hours in an oral glucose tolerance test
3. Blood Pressure — Diastolic blood pressure (mm Hg)
4. SkinThickness — Triceps skin-fold thickness (mm)
5. Insulin — Two hours of serum insulin (mu U/ml)
6. BMI — Body mass index (weight in kg/(height in m)²)
7. Diabetes Pedigree Function — Diabetes pedigree function
8. Age — Age in years
9. Outcome — Class variable (indicates whether the patient is diabetic or not)
 and available on the UCI Machine Learning Repository. The dataset includes data from 768 females of at least 21 years old of Pima Indian heritage. The datasets consist of several medical predictor variables and one target variable (whether the person has diabetes).

## Objective
The primary goal of this project is to build a predictive model that can identify the likelihood of a patient having diabetes based on certain diagnostic measurements.



## Technologies Used
--> Python
--> pandas - for dataset handling
--> numpy - for numerical computations
--> scikit-learn - for machine learning modelling and processing
--> matplotlib, seaborn - for data visualization



## Methodology
The project starts with data exploration where I get a sense of the data, its structure, and some summary statistics. Next, I perform data cleaning and preprocessing, which includes dealing with missing values and categorical data. Then I perform some exploratory data analysis. I use different visualizations to understand the relationship between different variables. After that, I prepare the data for machine learning models, split it into training and test datasets. Then, I use different classification algorithms to classify whether the person has diabetes or not. Finally, I evaluate each model and conclude with the model that performs best.



## Results
The results of our machine learning models can be summarized as follows:

### Classification Models:
After assessing the performance of various classification models - Logistic Regression, Support Vector Machine (SVM), and Naive Bayes - we found that Logistic Regression performed the best on our dataset. It achieved the highest Area Under the Curve (AUC) of 0.752 and an accuracy of 0.83. The model also had high precision, indicating a lower number of false positives.

The SVM model also performed well, achieving an accuracy of 0.83. However, its AUC was slightly lower at 0.737. The Naive Bayes model, while having a higher recall, which suggests a better identification of true positives, lagged in other metrics like accuracy and AUC.

However, the choice of model can depend on the specific requirements of the application, as other factors like computational complexity, interpretability, and scalability can also influence the decision.

### Regression Models:
Our evaluation of regression models - Linear Regression and Random Forest - revealed that Linear Regression performed slightly better on our dataset. It had lower Mean Absolute Error (MAE) and Mean Squared Error (MSE) values, indicating more accurate predictions on average. Furthermore, its R-squared value was higher, suggesting a better fit to the data.

In conclusion, the performance of a machine learning model can often be enhanced through methods like feature engineering(choosing and modifying the input variables to produce fresh features that enhance the performance of the model. Techniques like normalization, scaling, and dimensionality reduction can be used in this.), cross-validation(by dividing the data into training and validation sets, cross-validation is a technique used to evaluate a model's performance. As a result, overfitting is less likely to occur and the model's performance is more precisely estimated.), appropriate algorithm selection(The effectiveness of your classification model can be greatly impacted by selecting the appropriate algorithm for your situation. Logistic regression, decision trees, random forests, and support vector machines are common classification algorithms (SVMs).), and data preprocessing(The accuracy of the model can be increased by preprocessing the data before feeding it to it. To make sure the data is in the right format and range, methods including normalization, data scaling, and data cleaning can be utilized.)


## Installation and Usage
To clone this repository, you can run git clone <repository_link>.

You will need to install Python along with some packages: pandas, numpy, scikit-learn, matplotlib, and seaborn. These can be installed via pip using the command: pip install pandas numpy scikit-learn matplotlib seaborn'


## Contributing
Contributions are welcome. 
