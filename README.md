# diabetes_prediction
Diabetes prediction is important because it helps us identify people who are at risk of developing the disease. By accurately predicting diabetes, we can take early action to prevent or manage it effectively. This can lead to better health outcomes and help people make healthier choices.

### Problem Statement :-
The main goal of this diabetes prediction project is to create a machine learning model that can accurately predict the chances of individuals developing diabetes. We will analyze data related to their health and lifestyle factors to build a model that can estimate the risk of diabetes with great accuracy.

### Dataset Description :-
The dataset used for the Diabetes Prediction project was sourced from Kaggle, a popular platform for data science and machine learning enthusiasts.

## Attribute Information :-
The dataset for this diabetes prediction project contains the following columns:

#### 1.Pregnancies: This column represents the number of times a person has been pregnant.
#### 2.Glucose: This column represents the blood sugar level (glucose concentration) measured in the person’s body.
#### 3.Blood Pressure: This column represents the blood pressure level of the person in millimeters of mercury (mmHg).
#### 4.Skin Thickness: This column represents the skin thickness measured in millimeters.
#### 5.Insulin: This column represents the insulin level in the person’s body measured in milli-international units per milliliter (mu/ml).
#### 6.BMI (Body Mass Index): This column represents the individual’s body mass index, which is a measure of body fat based on height and weight.
#### 7.DiabetesPedigree: This column represents the diabetes pedigree function, which provides an indication of the genetic influence of diabetes based on family history.
#### 8.Age: This column represents the age of the person in years.
#### 9.Outcome: This column indicates the presence or absence of diabetes, where 1 represents the presence and 0 represents the absence.

#### dataset link: https://www.kaggle.com/datasets/saurabh00007/diabetescsv?source=post_page-----b02a1574a73f--------------------------------


### Data Preprocessing:
Before we could start building the model, we had to preprocess the data. This involved checking for missing values, removing any unnecessary features, and normalizing the data. We used pandas and scikit-learn libraries to perform these tasks.

### Exploratory Data Analysis:
Next, we performed some exploratory data analysis to gain insights into the relationships between the different features and the target variable, i.e., diabetes. We used the matplotlib and Seaborn libraries to visualize the data and understand any patterns or correlations in the dataset. We also computed some summary statistics to understand the central tendency and variability of the data.

### Model Building:
With the preprocessed data and insights from exploratory data analysis, we started building the machine learning model. We chose to use a logistic regression model since it's a simple and powerful algorithm for predicting continuous values. We used scikit-learn to split the dataset into training and testing sets, fit the model to the training data, and evaluate its performance on the testing data. We also tuned the hyperparameters of the model to improve its accuracy.

### Results:
Our logistic regression model was able to predict the diabetes with an accuracy of 0.79. Although the accuracy is not perfect, it's still a good starting point for further analysis and improvement.


### Conclusion :
We explored the dataset, built our model using the Logistic Regression, and evaluated its performance using various metrics. The Logistic regression gave an accuracy score of 79%, indicating that it can be used to predict the likelihood of developing diabetes early.


#### Linkedin: https://www.linkedin.com/in/neeraj-kumar-sharma-a4734b280/