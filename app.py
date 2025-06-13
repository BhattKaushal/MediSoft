import streamlit as st
import joblib
import numpy as np
from PIL import Image
import requests

if 'USER_DB' not in st.session_state:
    st.session_state.USER_DB = {}
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

@st.cache_resource
def load_model():
    return joblib.load("CNN_model.pkl")

model = load_model()

class_labels = ['Melanoma', 'Melanocytic Nevi']

menu = ["Login", "Create Account", "Image Classifier", "AI Chatbot"]
choice = st.sidebar.selectbox("Navigation", menu)

def authenticate_user(username, password):
    return username in st.session_state.USER_DB and st.session_state.USER_DB[username] == password

def create_account(username, password):
    if username in st.session_state.USER_DB:
        return False
    st.session_state.USER_DB[username] = password
    return True

if choice == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.error("Invalid username or password")

elif choice == "Create Account":
    st.title("Create Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        if create_account(new_username, new_password):
            st.success("Account created successfully! Please log in.")
        else:
            st.error("Username already exists.")

elif choice == "Image Classifier":
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to access the classifier.")
    else:
        st.title("Image Classifier")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            img_resized = image.resize((64, 64))
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            st.write("Image shape after preprocessing:", img_array.shape)
            st.write("Min/Max pixel values:", img_array.min(), img_array.max())
            st.image(img_resized, caption="Preprocessed Image", width=150)

            if st.button("Classify"):
                prediction_raw = model.predict(img_array)
                st.write("Raw prediction array:", prediction_raw)

                pred_value = prediction_raw[0][0]
                st.write("Prediction value (single float):", pred_value)

                if pred_value >= 0.5:
                    predicted_class = class_labels[1]
                    confidence = pred_value
                else:
                    predicted_class = class_labels[0]
                    confidence = 1 - pred_value

                confidence *= 100
                st.success(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")

        st.button("Logout", on_click=lambda: st.session_state.update({'logged_in': False, 'username': ""}))

elif choice == "AI Chatbot":
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in to access the chatbot.")
    else:
        st.title("Skin Disease Assistant Chatbot 🤖")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        openrouter_api_key = st.text_input("Enter your OpenRouter API Key", type="password")
        user_input = st.text_input("You:", key="chat_input")

        if openrouter_api_key and user_input:
            st.session_state.chat_history.append(("user", user_input))

            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "user", "content": msg} if sender == "user" else {"role": "assistant", "content": msg}
                    for sender, msg in st.session_state.chat_history
                ]
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]
                st.session_state.chat_history.append(("bot", reply))
            else:
                reply = "Sorry, something went wrong with the chatbot."
                st.session_state.chat_history.append(("bot", reply))

        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"**You:** {message}")
            else:
                st.markdown(f"**Bot:** {message}")
