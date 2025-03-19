import streamlit as st
import requests
import json
from ultralytics import YOLO
from PIL import Image

# Customizing the app theme
st.set_page_config(
    page_title="PPE Compliance Checker",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar styling
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #D8BFD8;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] .stCheckbox label {
        color: black !important;
    }
    </style>
"""

# Main page background styling
main_bg_style = """
    <style>
    .stApp {
        background-color: #FFFACD;
    }
    </style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)
st.markdown(main_bg_style, unsafe_allow_html=True)

# Langflow API Configuration
BASE_API_URL = "https://0f08-175-139-159-165.ngrok-free.app"
FLOW_ID = "28d0067f-0cad-4e4b-b72c-88520ce3261f"

# Function to send data to Langflow
def send_to_langflow(detected_ppe):
    payload = {"input_value": json.dumps({"detected_ppe": detected_ppe}), "output_type": "chat", "input_type": "chat"}
    response = requests.post(f"{BASE_API_URL}/api/v1/run/{FLOW_ID}", json=payload)
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Langflow."}

# Function to chat with bot
def chat_with_bot(user_input):
    payload = {"input_value": user_input, "output_type": "chat", "input_type": "chat"}
    response = requests.post(f"{BASE_API_URL}/api/v1/run/{FLOW_ID}", json=payload)
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Langflow."}

# Streamlit App
def main():
    st.title("🦺 PPE Compliance Checker 👷‍♂️")
    st.write("📸 Upload or capture an image to check PPE compliance.")
    
    # Sidebar options
    with st.sidebar:
        st.header("Quick Questions")
        sample_questions = [
            "What is PPE compliance?",
            "Why is wearing a hardhat important?",
            "What happens if PPE is missing?",
            "How does the AI detect PPE?"
        ]
        selected_question = st.selectbox("Choose a question", sample_questions)
        if st.button("Ask Bot"):
            st.session_state.messages.append({"role": "user", "content": selected_question})
            response = chat_with_bot(selected_question)
            bot_reply = response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "No response")
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    
        enable_camera = st.checkbox("Enable camera")
        picture = st.camera_input("Take a picture", disabled=not enable_camera)
        uploaded_file = st.file_uploader("Or upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    image = None
    if picture:
        image = Image.open(picture)
    elif uploaded_file:
        image = Image.open(uploaded_file)

    if image is not None:
        st.image(image, caption="Selected Image", use_container_width=True)
        model = YOLO("Model/best.pt")
        results = model(image)

        ppe_classes = ["hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"]
        detected_ppe = [ppe_classes[int(box.cls[0])] for box in results[0].boxes]

        st.write("### Detected PPE Items:")
        no_ppe_detected = any(item in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"] for item in detected_ppe)

        if detected_ppe:
            for item in detected_ppe:
                if item in ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]:
                    st.error(f"❌ {item} detected! PPE violation!")
                else:
                    st.success(f"✅ {item}")
        
        if no_ppe_detected:
            st.error("❌ No PPE detected! Please wear appropriate safety gear.")
        elif not detected_ppe:
            st.warning("⚠️ No PPE detected.")
        
        response = send_to_langflow(detected_ppe)
        st.write("### Compliance Result:")
        st.info(response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "No response"))
        
        st.image(results[0].plot(), caption="Detection Results")
    
    # Chatbot Section
    st.write("\n---")
    st.write("## 💬Chat with the PPE Compliance Bot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖"):
            st.markdown(content)
    
    if user_query := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(user_query)
        
        with st.chat_message("assistant", avatar="🤖"):
            response = chat_with_bot(user_query)
            bot_reply = response.get("outputs", [{}])[0].get("outputs", [{}])[0].get("results", {}).get("message", {}).get("text", "No response")
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

if __name__ == "__main__":
    main()
