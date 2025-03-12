#Linear Regression App with Streamlit UI Code
# The code snippet below is the Streamlit UI code for the Linear Regression app. This code will be saved in a file named src/app_LinearRegression.py.
# Save the following code in the src/app_LinearRegression.py file:


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the web app
st.title('Simple Streamlit App')

# Text input to take a parameter from the user
user_input = st.text_input("Enter a number to control sine wave frequency:")

# Default frequency
if user_input:
    try:
        frequency = float(user_input)
    except ValueError:
        st.write("Please enter a valid number.")
        frequency = 1.0
else:
    frequency = 1.0

# Create data for the sine wave
x = np.linspace(0, 10, 100)
y = np.sin(frequency * x)

# Plot the sine wave
plt.plot(x, y)
plt.title(f"Sine Wave with Frequency {frequency}")
plt.xlabel("X")
plt.ylabel("Y")

# Display the plot
st.pyplot(plt)

# Display the user input
st.write(f"You entered a frequency of {frequency}")
