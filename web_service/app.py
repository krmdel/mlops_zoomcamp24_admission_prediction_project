import streamlit as st
import pandas as pd
import requests
import io

# Title of the web app
st.title("Admission Prediction App")

# Option to either input the path or upload a file
st.subheader("Upload a CSV file or Enter CSV file path")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If the file is uploaded, process the uploaded file
if uploaded_file is not None:
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        # Convert the dataframe to CSV format as a string
        csv_data = df.to_csv(index=False)
        st.success("CSV file uploaded successfully.")
        
        # Send the CSV content to the Flask API
        st.write("Processing the file...")

        api_url = "http://localhost:9696/predict"
        response = requests.post(api_url, files={"file": io.StringIO(csv_data)})

        if response.status_code == 200:
            response_data = response.json()

            if not response_data.get("predictions"):
                st.error("No valid patient data found in the CSV file.")
            else:
                results = response_data["predictions"]
                s3_path = response_data.get("s3_url", "")

                # Dropdown menu to select a patient
                patient_ids = [result["patient_id"] for result in results]
                selected_patient_id = st.selectbox("Select Patient ID", patient_ids)

                # Find the selected patient's result
                prediction_result = next(result for result in results if result['patient_id'] == selected_patient_id)

                # Display the prediction result
                st.subheader("Prediction Result")
                st.write(f"Patient ID: {prediction_result['patient_id']}")
                st.write(f"Admission Prediction: {prediction_result['admission']}")

                # Display patient info
                st.subheader("Patient Information")
                for key, value in prediction_result.items():
                    if key != 'admission':
                        st.write(f"{key}: {value}")

                # Display the S3 path
                if s3_path:
                    st.success(f"Results have been saved into {s3_path} successfully.")
                else:
                    st.warning("S3 path not returned by the API.")

        else:
            st.error(f"Error from Flask API: {response.json().get('error')}")

    except Exception as e:
        st.error(f"Error processing the uploaded file: {e}")

else:
    st.write("Please upload a CSV file to proceed.")
