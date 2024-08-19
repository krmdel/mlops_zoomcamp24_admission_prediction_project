import streamlit as st
import pandas as pd
import requests
import io

# Title of the web app
st.title("Admission Prediction App")

# Subheader to prompt the user to upload a file or enter a file path
st.subheader("Upload a CSV file or Enter CSV file path")

# File uploader widget to allow the user to upload a CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If the user has uploaded a file, process it
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Convert the DataFrame back to CSV format as a string for API transmission
        csv_data = df.to_csv(index=False)
        st.success("CSV file uploaded successfully.")
        
        # Inform the user that the file is being processed
        st.write("Processing the file...")

        # Define the API URL where the CSV data will be sent for predictions
        api_url = "http://localhost:9696/predict"
        
        # Send the CSV data to the Flask API as a POST request
        response = requests.post(api_url, files={"file": io.StringIO(csv_data)})

        # Check if the API request was successful
        if response.status_code == 200:
            # Parse the JSON response from the API
            response_data = response.json()

            # Check if predictions were returned by the API
            if not response_data.get("predictions"):
                st.error("No valid patient data found in the CSV file.")
            else:
                # Extract predictions and optional S3 path from the API response
                results = response_data["predictions"]
                s3_path = response_data.get("s3_url", "")

                # Create a dropdown menu to allow the user to select a patient by ID
                patient_ids = [result["patient_id"] for result in results]
                selected_patient_id = st.selectbox("Select Patient ID", patient_ids)

                # Find the selected patient's prediction result
                prediction_result = next(result for result in results if result['patient_id'] == selected_patient_id)

                # Display the prediction result for the selected patient
                st.subheader("Prediction Result")
                st.write(f"Patient ID: {prediction_result['patient_id']}")
                st.write(f"Admission Prediction: {prediction_result['admission']}")

                # Display additional patient information (excluding the admission prediction)
                st.subheader("Patient Information")
                for key, value in prediction_result.items():
                    if key != 'admission':
                        st.write(f"{key}: {value}")

                # If an S3 path was returned, inform the user where the results were saved
                if s3_path:
                    st.success(f"Results have been saved into {s3_path} successfully.")
                else:
                    st.warning("S3 path not returned by the API.")

        else:
            # Handle any errors returned by the Flask API
            st.error(f"Error from Flask API: {response.json().get('error')}")

    except Exception as e:
        # Handle any errors that occurred during the file processing
        st.error(f"Error processing the uploaded file: {e}")

else:
    # Prompt the user to upload a file if they haven't already done so
    st.write("Please upload a CSV file to proceed.")
