# Titanic Survivor Prediction Model Project
The object of this project is to predict the survival of passengers on the Titanic using machine learning. 
The dataset used for this project is the “titanic.csv” dataset (from Kaggle) which consists of 891 instances and 12 attributes. 
The attributes include PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked, and Survived. 
The Survived attribute is the target variable which indicates whether the passenger survived (1) or not (0).

## Data Processing
The first step of any machine learning project is to preprocess the data. In this project, I have loaded the dataset using pandas 
and created two variables - predictor (X) and target (y). I have dropped the Survived column from the predictor variable and used 
it as the target variable. I have handled missing values in the Age, Cabin, and Embarked columns. Categorical variables such as Sex 
and Embarked are encoded using one-hot encoding. Numerical features such as Age and Fare are scaled to improve model performance. 
Additionally, I split the data into training and testing sets using the train_test_split function from scikit-learn.

## Model Training
I have used the Random Forest Classifier algorithm to train the model. The Random Forest Classifier is an ensemble learning method 
that uses decision trees to make predictions. It works by constructing multiple decision trees during training and outputting the class 
that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. I have trained the model using 
the training data and calculated the accuracy of the model on the test data.

## Conclusion
In this project, I have successfully predicted the survival of Titanic passengers using machine learning. I have used the Random Forest Classifier algorithm. 
This project can be extended by using other machine learning algorithms and improving the feature engineering process.

## About Data
The data provided appears to be a tabular dataset with 12 columns and multiple rows, where each row represents a passenger and each column 
represents different attributes of that passenger. The attributes are described below:

PassengerId: Unique identifier for each passenger.
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
Name: Name of the passenger.
Sex: Gender of the passenger.
Age: Age of the passenger.
SibSp: Number of siblings/spouses aboard the Titanic.
Parch: Number of parents/children aboard the Titanic.
Ticket: Ticket number.
Fare: Passenger fare.
Cabin: Cabin number.
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
Survived: Survival status (0 = No, 1 = Yes).

## About Code
Here, we are reading the Titanic dataset from a CSV file using Pandas and storing it in a Pandas DataFrame called titanic.
Creating the predictor variable X by dropping the Survived column from the titanic DataFrame. Creating the target variable y 
by using the Survived column. Handling missing values in the Age, Cabin, and Embarked columns. Categorical variables such as Sex 
and Embarked are encoded using one-hot encoding. Numerical features such as Age and Fare are scaled to improve model performance.
Splitting the dataset into training and testing sets using the train_test_split() function from Scikit-learn. The training set contains 
80% of the data, and the testing set contains 20% of the data. The random_state parameter sets the seed value for the random number generator 
to ensure that we get the same results every time we run the code. Initializing a Random Forest Classifier model and fitting it to the training 
data using the fit() method. Using the trained model to predict the survival status for the testing data and calculating the accuracy of the model 
on the testing data using the accuracy_score() function from Scikit-learn. The accuracy score is printed to the console.
