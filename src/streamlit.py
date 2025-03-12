# app.py
import streamlit as st

# Title of the web app
st.title('Addition of Two Numbers')

# Take two inputs from the user
num1 = st.number_input('Enter the first number:', value=0)
num2 = st.number_input('Enter the second number:', value=0)

# Calculate the sum of both numbers
result = num1 + num2

# Display the result
st.write(f'The sum of {num1} and {num2} is: {result}')
