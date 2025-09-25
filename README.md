# Bengalore-_house_price-prediction
Bangalore House Price Prediction
This project is a machine learning model that predicts house prices in Bangalore, India. The model is trained on a dataset of real estate listings and is deployed as a simple, interactive web application using Streamlit.

Table of Contents
Overview

Features

Tech Stack

Project Structure

How to Run

Model Development Process

Overview
The primary goal of this project is to build an accurate and reliable house price prediction model. The project involves a complete machine learning workflow:

Data Cleaning & Preprocessing: The raw dataset is extensively cleaned, handling missing values, formatting inconsistencies, and removing outliers.

Feature Engineering: New features are derived, and existing ones are transformed to be suitable for a machine learning model.

Model Training: A Linear Regression model is trained on the processed data.

Deployment: The trained model is saved and integrated into a Streamlit web application, allowing users to get instant price predictions based on their inputs.

Features
Dynamic Price Prediction: Get real-time house price estimates based on input features.

User-Friendly Interface: A clean and simple web interface for easy interaction.

Comprehensive Data Cleaning: The model is built on a thoroughly cleaned and preprocessed dataset, ensuring more reliable predictions.

Reproducible Workflow: The entire process, from data cleaning to model training, is documented in the Jupyter Notebook.

Tech Stack
Programming Language: Python
-- Data Manipulation: Pandas, NumPy

Machine Learning: Scikit-learn

Web Framework: Streamlit

Development Environment: Jupyter Notebook

Project Structure
├── app.py                            # The Streamlit web application script
├── bangalore-house-price-prediction-using-ml.ipynb # Jupyter Notebook for data cleaning, model training, and evaluation
├── linear_regression_model.pkl       # Saved (pickled) trained Linear Regression model
├── label_encoder.pkl                 # Saved label encoder for location features
├── Bengaluru_House_Data.csv          # The original, raw dataset
├── df10.csv                          # The final, cleaned dataset used for training


How to Run
To run this project locally, please follow these steps:

1. Clone the repository:

git clone [https://github.com/your-username/bangalore-house-price-prediction.git](https://github.com/your-username/bangalore-house-price-prediction.git)
cd bangalore-house-price-prediction

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required libraries:
It's recommended to create a requirements.txt file with the following content:

pandas
numpy
scikit-learn
streamlit

Then, install them using pip:

pip install -r requirements.txt

4. Run the Streamlit application:

streamlit run app.py

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

Model Development Process
The model was developed in the bangalore-house-price-prediction-using-ml.ipynb notebook. The key steps were:

Data Loading: The Bengaluru_House_Data.csv dataset was loaded into a pandas DataFrame.

Data Cleaning:

Handled missing values in columns like bath and balcony.

Standardized the size column to a numerical bhk (Bedrooms, Hall, Kitchen) column.

Cleaned the total_sqft column by converting ranges (e.g., "1133 - 1384") into single numerical values.

Outlier Removal: Outliers were removed to improve the model's accuracy and generalization. This was done using domain knowledge, such as removing properties with an abnormally high price per square foot or an unusual number of bathrooms relative to bedrooms.

Feature Transformation:

The categorical location feature was converted into numerical data using one-hot encoding, creating a separate column for each location.

Model Training: A Linear Regression model was chosen and trained on the final, cleaned dataset (df10.csv).

Exporting Artifacts: The trained model and the label encoder were saved as .pkl files to be used by the Streamlit application.
