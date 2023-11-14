# # Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # import streamlit as st
# # import pandas as pd
# # from streamlit.logger import get_logger

# LOGGER = get_logger(__name__)


# def run():
#     st.set_page_config(
#         page_title="Hello",
#         page_icon="👋",
#     )

#     st.write("# ngehhh")

#     st.sidebar.success("Select a demo above.")

#     st.markdown(
#         """
#         Streamlit is an open-source app framework built specifically for
#         Machine Learning and Data Science projects.
#         **👈 Select a demo from the sidebar** to see some examples
#         of what Streamlit can do!
#         ### Want to learn more?
#         - Check out [streamlit.io](https://streamlit.io)
#         - Jump into our [documentation](https://docs.streamlit.io)
#         - Ask a question in our [community
#           forums](https://discuss.streamlit.io)
#         ### See more complex demos
#         - Use a neural net to [analyze the Udacity Self-driving Car Image
#           Dataset](https://github.com/streamlit/demo-self-driving)
#         - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
#     """
#     )


# if __name__ == "__main__":
#     run()

import streamlit as st
import joblib
from textblob import TextBlob

# Load the TF-IDF vectorizer and trained model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
logreg_model = joblib.load('your_trained_logreg_model.joblib')  # Replace with your trained logistic regression model

# Define a function to predict sentiment
def predict_sentiment(text):
    text_vectorized = tfidf_vectorizer.transform([text])
    prediction = logreg_model.predict(text_vectorized)
    return prediction

# Streamlit app
def main():
    st.title("Sentiment Analysis App")
    
    # User input
    user_input = st.text_area("Enter a movie review:")

    if st.button("Predict Sentiment"):
        if user_input:
            sentiment = predict_sentiment(user_input)

            if sentiment == 1:
                sentiment_label = "Positive"
            else:
                sentiment_label = "Negative"

            st.success(f"Predicted Sentiment: {sentiment_label}")

if __name__ == '__main__':
    main()

