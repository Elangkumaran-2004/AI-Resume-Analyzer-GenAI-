import streamlit as st
from utils import extract_text_from_pdf
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

genai.configure(api_key=api_key)

# -----------------------------
# Load Gemini Model
# -----------------------------
model = genai.GenerativeModel("gemini-flash-latest")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🤖 AI Resume Analyzer")

st.sidebar.markdown("## 🛠 Technologies Used")

st.sidebar.success("Python")
st.sidebar.success("Streamlit")
st.sidebar.success("Google Gemini AI")
st.sidebar.success("PyPDF2")

st.sidebar.markdown("---")

st.sidebar.info("Developed by Elang Kumaran")

# -----------------------------
# Title
# -----------------------------
st.title("📄 AI Resume Analyzer")

st.write("Upload your resume and get an AI-powered ATS analysis.")

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# -----------------------------
# Resume Upload
# -----------------------------
if uploaded_file is not None:

    resume_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📃 Resume Preview")

    with st.expander("View Resume"):
        st.write(resume_text)

    if st.button("🚀 Analyze Resume"):

        prompt = f"""
You are an ATS Resume Expert.

Analyze the resume and provide:

1. ATS Score out of 100
2. Resume Summary
3. Strengths
4. Weaknesses
5. Missing Technical Skills
6. Suggested Improvements
7. Recommended AI/ML Job Roles

Resume:

{resume_text}
"""

        try:
            with st.spinner("🔍 Analyzing Resume... Please wait..."):

                response = model.generate_content(prompt)

                if response is None:
                    st.error("❌ No response received from Gemini AI.")
                    st.stop()

                if hasattr(response, "text") and response.text:
                    result = response.text
                else:
                    st.error("❌ Gemini returned an empty response.")
                    st.stop()

            st.success("✅ Resume Analysis Completed!")

        except Exception as e:

            error = str(e)

            if "429" in error or "ResourceExhausted" in error:
                st.error("🚫 Gemini API quota exceeded.")
                st.info("Please wait a few minutes and try again.")

            elif "API_KEY" in error or "DefaultCredentialsError" in error:
                st.error("❌ Invalid or Missing Gemini API Key.")
                st.info("Check your .env file.")

            elif "404" in error:
                st.error("❌ Model not found.")
                st.info("Use gemini-2.5-flash")

            elif "Connection" in error or "Timeout" in error:
                st.error("🌐 Internet connection error.")
                st.info("Check your internet connection.")

            else:
                st.error("❌ Unexpected Error")
                st.exception(e)

            st.stop()

        # ATS Score
        score = 70

        try:
            match = re.search(
                r'ATS Score.*?(\d+)',
                result,
                re.IGNORECASE
            )

            if match:
                score = int(match.group(1))

        except:
            score = 70

        st.markdown("## 📊 Estimated ATS Score")

        st.progress(score)

        st.metric(
            "Estimated ATS Score",
            f"{score}/100"
        )

        st.markdown("---")

        st.markdown("## 🚀 Resume Improvement Suggestions")

        col1, col2 = st.columns(2)

        with col1:
            st.success("### 🔥 Add These Skills")

            st.markdown("""
- ✅ TensorFlow
- ✅ PyTorch
- ✅ LangChain
- ✅ Hugging Face
- ✅ FastAPI
- ✅ Docker
- ✅ Prompt Engineering
- ✅ ChromaDB
- ✅ FAISS
- ✅ NLP
- ✅ Deep Learning
""")

        with col2:
            st.info("### 💼 Recommended Roles")

            st.markdown("""
- 🤖 AI Engineer
- 🧠 Machine Learning Engineer
- 💬 Generative AI Engineer
- 📊 Data Scientist
- 📈 Data Analyst
- 🔍 NLP Engineer
- 👨‍💻 AI/ML Intern
""")

        st.markdown("---")

        st.markdown("## 🤖 AI Resume Analysis")

        st.markdown(
            f"""
<div style="
background-color:#1e1e1e;
padding:25px;
border-radius:15px;
border:2px solid #4CAF50;
font-size:17px;
line-height:1.8;
">

{result}

</div>
""",
            unsafe_allow_html=True
        )