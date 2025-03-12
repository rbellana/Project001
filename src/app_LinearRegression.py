# Linear Regression App with Streamlit UI Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# File path for the dataset
file_path = '/home/ram/Projects/Project001_Sandbox/Project001/src/BostonHousing.csv'

# Load the dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, header=0)
    return df

# Preprocess data
def preprocess_data(df):
    # Check for missing values
    st.write("Data Info:")
    st.write(df.info())
    
    # Handle missing values (if any)
    df = df.dropna()  # Dropping rows with missing values. You can opt to impute or handle them differently.
    
    # Check if 'medv' exists in the dataset
    if 'medv' not in df.columns:
        st.error("The column 'medv' is not found in the dataset.")
        return None, None
    
    # Split features and target variable
    X = df.drop(columns=['medv'])  # Features
    y = df['medv']  # Target variable

    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Train a linear regression model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f'Mean Squared Error (MSE): {mse}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse}')
    st.write(f'R2 Score: {r2}')
    
    # Plotting the predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title('Predicted vs Actual')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    st.pyplot()

# Streamlit UI
def main():
    # Load data
    df = load_data(file_path)
    
    st.title('Boston Housing Linear Regression')
    st.write('This app uses Linear Regression to predict housing prices from the Boston dataset.')
    
    # Show data preview
    if st.checkbox('Show raw data'):
        st.write(df.head())
    
    # Preprocess data
    X_scaled, y = preprocess_data(df)
    
    if X_scaled is None or y is None:
        return  # Exit if 'medv' column is missing
    
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
