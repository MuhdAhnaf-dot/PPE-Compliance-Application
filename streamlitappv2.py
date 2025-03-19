#%%
import streamlit as st
import requests
import json
import logging
from ultralytics import YOLO
from PIL import Image

# Langflow API Configuration
BASE_API_URL = "https://0f08-175-139-159-165.ngrok-free.app"
FLOW_ID = "28d0067f-0cad-4e4b-b72c-88520ce3261f"
ENDPOINT = ""  # Replace with your actual Flow ID

# Function to send data to Langflow
def send_to_langflow(detected_ppe):
    payload = {
        "input_value": json.dumps({"detected_ppe": detected_ppe}),
        "output_type": "chat",
        "input_type": "chat"
    }

    response = requests.post(f"{BASE_API_URL}/api/v1/run/{FLOW_ID}", json=payload)

    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Langflow."}

# Function to send chat messages to Langflow
def chat_with_bot(user_input):
    payload = {
        "input_value": user_input,
        "output_type": "chat",
        "input_type": "chat"
    }
    response = requests.post(f"{BASE_API_URL}/api/v1/run/{FLOW_ID}", json=payload)
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Langflow."}

# Streamlit App
def main():
    st.title("🦺 PPE Compliance Checker 👷‍♂️")
    st.write("📸 Upload or capture an image to check PPE compliance.")

    # Sidebar options: Upload image OR use camera
    with st.sidebar:
        enable_camera = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable_camera)
        uploaded_file = st.file_uploader("Or upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

    # Select the image source (Camera or Upload)
    image = None
    if picture:
        image = Image.open(picture)
    elif uploaded_file:
        image = Image.open(uploaded_file)

    if image is not None:
        st.image(image, caption="Selected Image", use_container_width=True)

        # Load YOLO model 
        model = YOLO("Model/best.pt")
        results = model(image)

        # Extract detected items
        ppe_classes = ["hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
                       "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"]
        detected_ppe = [ppe_classes[int(box.cls[0])] for box in results[0].boxes]

        # Show detections
        st.write("### Detected PPE Items:")
        if detected_ppe:
            for item in detected_ppe:
                st.write(f"✅ {item}")
        else:
            st.write("⚠️ No PPE detected.")

        # Send detections to Langflow
        response = send_to_langflow(detected_ppe)
        st.write("### Compliance Result:")
        st.write(response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "No response"))

        #Show detection result image
        st.image(results[0].plot(), caption="Detection Results")

    # Chatbot Section
    st.write("\n---")
    st.write("## 🧑‍💼💬🤖Chat with the PPE Compliance Bot")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages with proper alignment
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)

    # Chat input
    if user_query := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(f"<div style='text-align: right;'>{user_query}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant", avatar="🤖"):
            response = chat_with_bot(user_query)
            bot_reply = response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "No response")
            st.markdown(f"<div style='text-align: left;'>{bot_reply}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

if __name__ == "__main__":
    main()
# %%%%
