import streamlit as st
import pickle
import pandas as pd
from PIL import Image, ImageDraw
import io
import base64


# ---------- BACKGROUND IMAGE ----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_bg_from_local("bg.jpeg")

# load model file
model = pickle.load(open('diabetes_Prediction.save', 'rb'))
scaler = pickle.load(open('diabetes_scaler.save', 'rb'))
le = pickle.load(open('diabetes_le.save', 'rb'))
le1 = pickle.load(open('diabetes_le1.save', 'rb'))


def change_page(page_name):
    st.session_state.page = page_name


def home_page():
    st.title(':blue[DIABETES PREDICTION]')

    st.markdown("""
    Diabetes is a chronic medical condition that occurs when the body cannot properly regulate blood sugar (glucose) levels. It can result from insufficient insulin production or the body's inability to effectively use insulin. Proper diet, exercise, medication, and regular monitoring help manage diabetes and reduce health complications.
    """)

    img = Image.open('diabetes.jpeg')
    st.image(img, width=500)

    st.button("Go to Next Page", on_click=change_page, args=("prediction",))


def prediction_page():
    st.title(':blue[DIABETES PREDICTION]')

    st.button("Back to Home", on_click=change_page, args=("home",))

    st.write("Please enter your health details below:")

    gender = st.selectbox('Gender', ['Male', 'Female'])

    age = st.number_input('Age', 0)

    hypertension = st.radio('Hypertension', ['YES', 'NO'])
    hypertension = 1 if hypertension == 'YES' else 0

    heart_disease = st.radio('Heart Disease', ['YES', 'NO'])
    heart_disease = 1 if heart_disease == 'YES' else 0

    smoking_history = st.selectbox(
        'Smoking History',
        ['never', 'No Info', 'current', 'former', 'ever', 'not current']
    )

    bmi = st.number_input('BMI', 0.0, 100.0)

    HbA1c_level = st.number_input('HbA1c Level', 0.0, 10.0)

    blood_glucose_level = st.number_input('Blood Glucose Level', 0)

    pred = st.button('RESULT')

    if pred:

        gen = le1.transform([gender])[0]
        smok = le.transform([smoking_history])[0]

        features = [[
            gen,
            age,
            hypertension,
            heart_disease,
            smok,
            bmi,
            HbA1c_level,
            blood_glucose_level
        ]]

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)

        result_text = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes"

        if prediction[0] == 1:
            st.error(result_text)
        else:
            st.success(result_text)

        # ---------------- RESULT DATAFRAME ---------------- #

        result_df = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Hypertension': [hypertension],
            'Heart Disease': [heart_disease],
            'Smoking History': [smoking_history],
            'BMI': [bmi],
            'HbA1c Level': [HbA1c_level],
            'Blood Glucose Level': [blood_glucose_level],
            'Prediction': [result_text]
        })

        st.dataframe(result_df)

        # ---------------- CSV DOWNLOAD ---------------- #

        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='prediction_result.csv',
            mime='text/csv',
        )

        # ---------------- PNG IMAGE CREATION ---------------- #

        img = Image.new('RGB', (600, 400), color='white')

        draw = ImageDraw.Draw(img)

        text = f"""
        DIABETES PREDICTION RESULT

        Gender : {gender}
        Age : {age}
        Hypertension : {hypertension}
        Heart Disease : {heart_disease}
        Smoking History : {smoking_history}
        BMI : {bmi}
        HbA1c Level : {HbA1c_level}
        Blood Glucose Level : {blood_glucose_level}

        Prediction : {result_text}
        """

        draw.text((20, 20), text, fill='black')

        # Save image into memory
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        
        st.image(img, caption="Prediction Result")

        st.download_button(
            label="Download PNG",
            data=buffer.getvalue(),
            file_name="prediction_result.png",
            mime="image/png"
        )


def main():

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()

    elif st.session_state.page == "prediction":
        prediction_page()


if __name__ == '__main__':
    main()