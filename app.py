import streamlit as st
import numpy as np
import pandas as pd

# Function to generate coefficients in matrix
def coefficent(E, N, n):
    columns = [np.ones(len(E))]
    for i in range(1, n + 1):
        columns.append(E ** i)
    for i in range(1, n + 1):
        columns.append(N ** i)
    for i in range(1, n):
        for j in range(1, -i + 1):
            columns.append((E ** i) * (N ** j))
    X = np.column_stack(columns)
    return X

# Function to get Z values for E, N, UN, n
def getZ(E, N, UN, n):
    X = coefficent(E, N, n)
    AT = np.transpose(X)
    ATX = np.dot(AT, X)
    ATXI = np.linalg.inv(ATX)
    Y = np.dot(ATXI, AT)
    Z = np.dot(Y, UN)
    return Z

# Main function
def main():
    st.title("Least Squares Surface Fitting")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload E,N,U CSV data file ", type=["csv"])
    
    if uploaded_file is not None:
        # Read CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

         # Display the DataFrame
        st.write(df.style.set_table_styles([{'selector': 'table', 'props': [('width', '800px')]}]))


        # Extract columns E, N, U as NumPy arrays
        E = df['E'].values
        N = df['N'].values
        UN = df['U'].values

        # Define the order 'n' for the polynomial
        n = st.selectbox("Select the order of the polynomial", options=[1,2, 3, 4, 5,6,7])

        # Calculate Z values
        Z = getZ(E, N, UN, n)

        # Take user input for E and N
        En = st.number_input("Enter E:")
        Nn = st.number_input("Enter N:")

        # Calculate u value
        p = coefficent(np.array([En]), np.array([Nn]), n)
        u = np.dot(p[0], Z)

        # Display the calculated 'u' value
        st.write(f"Calculated 'u' value: {u}")

if __name__ == "__main__":
    main()
