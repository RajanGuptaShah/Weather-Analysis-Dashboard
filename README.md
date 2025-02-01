Weather Prediction Dashboard



Table of Contents
Introduction
Features
Technologies Used
Dataset Information
Data Preprocessing
Model Evaluation
Shiny Dashboard
Project Setup
Usage
Visualization Screenshots
Future Scope


Introduction:

The Weather Prediction Dashboard is a comprehensive data-driven web application designed to predict weather conditions using multiple machine learning models. The application includes interactive visualizations and comparisons of model performance and predictions.


Features:

Data Preprocessing: Cleaning and normalization of weather data for accurate predictions.


Machine Learning Models: 

Evaluation using K-Nearest Neighbors (KNN), Naive Bayes, Decision Tree, and Support Vector Machine (SVM).


Model Performance Comparison: 

Accuracy comparison and confusion matrix visualizations for all models.


Interactive Dashboard: 

Developed using Shiny to dynamically display model accuracies, confusion matrices, and prediction comparisons.


Technologies Used:

Programming Language: R
Libraries: Data Processing: dplyr, caret, e1071
Machine Learning: class, rpart
Visualization: ggplot2, rpart.plot
Dashboard: shiny


Dataset Information:

The project uses a weather dataset containing features such as precipitation type, temperature, humidity, and wind speed. The dataset was preprocessed to remove null values and duplicate entries.


Data Fields:

Precip.Type: Type of precipitation (e.g., Sunny, Rain, Snow).
Temperature: Temperature in Celsius.
Humidity: Measured humidity levels.
Wind Speed: Wind speed in km/h.


Data Preprocessing:

Data Cleaning: Removed duplicate records and handled missing values by assigning Sunny to null precipitation types.

Feature Scaling: Normalized Temperature, Humidity, and Wind Speed using the scale() function.

Partitioning: Split the dataset into training (80%) and testing (20%) sets using the caret library.

Model Evaluation
K-Nearest Neighbors (KNN)
Achieved high prediction accuracy with k=3.

Naive Bayes
Performed efficiently with categorical data features.

Decision Tree
Visual representation using rpart.plot.

Support Vector Machine (SVM)
Used a linear kernel for classification.

Shiny Dashboard
The Shiny dashboard provides the following features:

Model Accuracy Visualization: Bar plots comparing the accuracy of all models.

Confusion Matrix Visualization: Interactive plots showing confusion matrices for selected models.

Prediction Comparisons: Tabular comparison between predicted and actual precipitation types.

Project Setup
Prerequisites
R and RStudio installed.

Required R packages: caret, class, e1071, rpart, rpart.plot, shiny, ggplot2.

Installation Steps
Clone this repository:

git clone https://github.com/RajanGuptaShah/Weather-Analysis-Dashboard.git
cd Weather-Analysis-Dashboard
Install the required R packages:

install.packages(c("caret", "class", "e1071", "rpart", "rpart.plot", "shiny", "ggplot2"))
Run the Shiny application:

shiny::runApp()
Usage
Select the desired machine learning model from the sidebar.

Click "Show Accuracy" to view the model accuracy plot.

Click "Show Confusion Matrix" to view the confusion matrix plot.

View prediction comparison tables for detailed analysis.

Visualization Screenshots
[Add screenshots showcasing the dashboard and model visualizations]

Future Scope
Integration with real-time weather data APIs.

Deployment of the dashboard on a web server.

Enhancement with additional machine learning models.

User account management for custom data analysis.
