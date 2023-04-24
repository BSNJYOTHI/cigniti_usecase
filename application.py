import streamlit as st
import requests
import json

# Define function to call API and get prediction
def predict_defect(description: str):
    

    # API endpoint
    url = "http://localhost:8000/predict-defect"

    # Define headers and data for API request
    headers = {"Content-Type": "application/json"}
    data = {"description": description}

    # Make API request and get response
    response = requests.get(url, headers=headers, params=data)
    response_json = json.loads(response.text)

    # Return prediction
    return response_json

# Define Streamlit app
def main():
    # Set page title
    st.set_page_config(page_title="Defect Predictor")

    # Define page layout
    st.title("Defect Predictor")

    # Define input field and submit button
    description = st.text_input("Enter defect description:", "")
    submitted = st.button("Submit")

    # Make prediction when submit button is clicked
    if submitted:
        # Check if input field is not empty
        if description != "":
            # Call predict_defect function to get prediction
            result = predict_defect(description)

            # Display prediction results
            st.write("Predicted Defect Status: ", result["Predicted Defect Status"])
            st.write("Confidence Score: ", result["Confidence Score"])

            # Check if reason prediction is available
            if "reason prediction" in result:
                st.write(result["reason prediction"])

# Run Streamlit app
if __name__ == "__main__":
    main()






