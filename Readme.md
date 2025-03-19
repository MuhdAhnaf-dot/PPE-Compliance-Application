**PPE Compliance Chatbot**

🧑‍💻 About

The PPE Compliance Chatbot is a deep learning-powered web application that allows users to upload images and snap an image to detect the PPE wearing by person in the image. The chatbot uses YOLOv8 for object detection, Langflow for chatbot interaction, and Streamlit for the user interface. The project is deployed on Streamlit Cloud, making it publicly accessible to anyone.

🎯 Features

✅ Upload Images: Users can upload images of workers wearing PPE.
⏰ Real-time PPE Detection: The chatbot identifies and verifies compliance with required PPE.
📋 Compliance Check: Determines whether workers meet safety regulations.
🌐 Accessible Anywhere: Hosted on Streamlit Cloud for public access.

🛠️ Technical Workflow

Deep Learning: YOLOv8 
Chatbot Framework: Langflow for user interaction
Deployment: Streamlit Cloud


📁 Project Structure

📦 ppe-compliance-chatbot
 ┣ 📂 Models                # Model saved
 ┣ 📂 images                # Sample images for testing
 ┣ 📜 streamlitappv2.py     # Main Streamlit app
 ┣ 📜 requirements.txt      # Dependencies for deployment
 ┣ 📜 Readme.md             # Project documentation (this file)

⛑️ PPE Classes = ["hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest","Person", "Safety Cone", "Safety Vest", "Machinery", "Vehicle"]

Example of quick question:
1. What is the meaning of PPE?
2. What is the purpose using PPE?
3. What can you detect from that image?

📑 Dataset
From: https://www.kaggle.com/code/hinepo/yolov8-finetuning-for-ppe-detection

🌐🌐 The link for a user to try in the streamlit cloud:
https://ppe-compliance-application-ptbkhfpj6aysjkl7ahbxqt.streamlit.app/