"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/
	

"""
import streamlit as st
import joblib
from PIL import Image

# Paths to models and vectorizer
vectorizer_path = "C:/Users/sthok/Desktop/my-streamlit-app/vectorizer.pkl"
logistic_regression_path = "C:/Users/sthok/Desktop/my-streamlit-app/logistic_regression.pkl"
naive_bayes_path = "C:/Users/sthok/Desktop/my-streamlit-app/naive_bayes.pkl"
gradient_boosting_path = "C:/Users/sthok/Desktop/my-streamlit-app/gradient_boosting.pkl"

# Load vectorizer and models
vectorizer = joblib.load(vectorizer_path)
logistic_regression_model = joblib.load(logistic_regression_path)
naive_bayes_model = joblib.load(naive_bayes_path)
gradient_boosting_model = joblib.load(gradient_boosting_path)

# Dictionary for mapping predictions to categories
category_dict = {0: "Business", 1: "Technology", 2: "Sports", 3: "Education", 4: "Entertainment"}

# The main function where we will build the actual app
def main():
    """News Article Category Classifier App with Streamlit """

    # Set the main title and subtitle
    st.title("NewsCat")

    # Creating sidebar with selection box - you can create multiple pages this way
    options = ["Home", "About Us", "Prediction", "EDA"]
    selection = st.sidebar.selectbox("Navigation", options)

    # Building out the Home page
    if selection == "Home":
        st.image("C:/Users/sthok/Desktop/AppLogo/logo.png", width=500)
        st.markdown("# Welcome to NewsCat!")
        st.markdown("## Your one-stop app for news article categorization")
        st.markdown("Use the sidebar to navigate through the app.")

    # Building out the "About Us" page
    if selection == "About Us":
        st.info("About Us")
        st.markdown("""
            Founded in 2023 by NewsCat Digitals

            **Mission Statement:**
            At NewsCat, our mission is to revolutionize how news articles are categorized and accessed. We aim to provide a seamless and efficient platform that automates the classification of news content, ensuring accuracy, reliability, and accessibility for our users worldwide.

            **Overview:**
            NewsCat is an innovative application designed to classify news articles into predefined categories such as Business, Technology, Sports, and more. Powered by advanced machine learning algorithms, NewsCat automates the categorization process, eliminating the need for manual sorting. This not only saves time and effort but also ensures consistent and reliable classification results. Whether you're a news enthusiast, researcher, or journalist, NewsCat provides a streamlined solution to access organized and relevant news content effortlessly.

            **Meet Our Team:**
            - **Founder and CEO:** Akhona Nzama
            - **Chief Technology Officer (CTO):** Britney Masalesa
            - **Lead Developer:** Jamie Hamann
            - **Project Manager:** Lindokuhle Mhlongo

            Join us as we continue to innovate and redefine how news is categorized and consumed in the digital age.
        """)

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        st.image("C:/Users/sthok/Desktop/pre.jpeg", width=700)  # Add image and set width
        st.markdown("### Enter the news article text below:")
        news_text = st.text_area("Enter Text", "Type here...")

        model_options = ["Logistic Regression", "Naive Bayes", "Gradient Boosting"]
        model_choice = st.selectbox("Choose Model", model_options)

        if st.button("Classify"):
            if news_text.strip() == "":
                st.error("Please enter text for classification")
            else:
                try:
                    # Transforming user input with vectorizer
                    vect_text = vectorizer.transform([news_text]).toarray()

                    # Load the chosen model
                    if model_choice == "Logistic Regression":
                        predictor = logistic_regression_model
                    elif model_choice == "Naive Bayes":
                        predictor = naive_bayes_model
                    elif model_choice == "Gradient Boosting":
                        predictor = gradient_boosting_model

                    # Make predictions
                    prediction = predictor.predict(vect_text)
                    category = category_dict.get(prediction[0], "Unknown")

                    # When model has successfully run, will print prediction
                    st.success(f"Text Categorized as: **{category}**")
                except Exception as e:
                    st.error(f"An error occurred during classification: {e}")

    # Building out the EDA page
    if selection == "EDA":
        st.info("Exploratory Data Analysis")
        st.markdown("""
            ### EDA is done to gain insights into the dataset and understand its characteristics. EDA helps in identifying patterns, trends, and relationships within the data.
        """)
        st.image("C:/Users/sthok/Desktop/appeda/bar.png", caption="Bar Chart", use_column_width=True)
        st.image("C:/Users/sthok/Desktop/appeda/cm.png", caption="Confusion Matrix", use_column_width=True)
        st.image("C:/Users/sthok/Desktop/appeda/dh.png", caption="Data Heatmap", use_column_width=True)
        st.image("C:/Users/sthok/Desktop/appeda/dl.png", caption="Data Lineplot", use_column_width=True)

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
