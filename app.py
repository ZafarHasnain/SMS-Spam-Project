import streamlit as st
import joblib

# 1. Load the files
model = joblib.load('trained_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# 2. Setup the App
st.title("📱 SMS Spam Classifier")
st.write("Enter a message to check if it's Spam or Ham.")

# 3. Input
sms_input = st.text_area("Message:")

if st.button("Check Message"):
    if sms_input:
        # Transform and Predict
        input_data = vectorizer.transform([sms_input])
        prediction = model.predict(input_data)[0]
        
        # --- DEBUGGING LINE (Optional: see result in terminal) ---
        print(f"Model predicted: {prediction}") 
        
        # 4. Display Result (Handles both 1/0 and 'spam'/'ham' labels)
        if prediction == 1 or prediction == 'spam':
            st.error(f"🚨 SPAM DETECTED! (Class: {prediction})")
        else:
            st.success(f"✅ Safe Message (Ham). (Class: {prediction})")
            
    else:
        st.warning("Please type something first.")