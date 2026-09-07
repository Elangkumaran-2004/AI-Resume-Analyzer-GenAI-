import streamlit as st
from utils import extract_text_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file")
    st.stop()

# -----------------------------
# Define Structured Output Schema
# -----------------------------
class ResumeAnalysis(BaseModel):
    ats_score: int = Field(description="ATS match score out of 100 based on formatting and skill presence")
    summary: str = Field(description="A concise executive summary of the resume")
    strengths: List[str] = Field(description="Top key strengths identified")
    weaknesses: List[str] = Field(description="Areas needing improvement")
    missing_skills: List[str] = Field(description="Critical missing technical skills for AI/ML roles")
    suggested_improvements: List[str] = Field(description="Actionable bullet points to optimize the resume")
    recommended_roles: List[str] = Field(description="Top 3 to 5 matching job titles")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.sidebar.title("🤖 AI Resume Analyzer")
st.sidebar.markdown("## 🛠 Technologies Used")
st.sidebar.success("Python & LangChain")
st.sidebar.success("Streamlit")
st.sidebar.success("Google Gemini 2.5")
st.sidebar.info("Developed by Elang Kumaran")

st.title("📄 AI Resume Analyzer (LangChain Powered)")
st.write("Upload your resume and get an AI-powered ATS analysis using LangChain orchestration.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

    with st.expander("📃 View Extracted Resume Text"):
        st.write(resume_text)

    if st.button("🚀 Analyze Resume"):
        try:
            with st.spinner("🔍 Running LangChain Analysis Pipeline..."):
                # Initialize LangChain LLM wrapper
                llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",  # ✅ Current active Flash model
                google_api_key=api_key,
                temperature=0.2
                )
                # Initialize Pydantic Parser to enforce valid JSON responses
                parser = PydanticOutputParser(pydantic_object=ResumeAnalysis)

                # Build Prompt Template
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert ATS (Applicant Tracking System) and Senior Technical Recruiter.\n{format_instructions}"),
                    ("human", "Analyze the following resume text:\n\n{resume_text}")
                ])

                # Construct LangChain Expression Language (LCEL) Chain
                chain = prompt_template | llm | parser

                # Execute Chain
                analysis: ResumeAnalysis = chain.invoke({
                    "resume_text": resume_text,
                    "format_instructions": parser.get_format_instructions()
                })

            st.success("✅ Analysis Completed via LangChain!")

            # -----------------------------
            # UI Render
            # -----------------------------
            st.markdown("## 📊 Estimated ATS Score")
            st.progress(analysis.ats_score / 100)
            st.metric("Estimated ATS Score", f"{analysis.ats_score}/100")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔥 Missing Technical Skills")
                for skill in analysis.missing_skills:
                    st.write(f"- ❌ {skill}")

            with col2:
                st.subheader("💼 Recommended Roles")
                for role in analysis.recommended_roles:
                    st.write(f"- 🤖 {role}")

            st.markdown("---")
            st.subheader("📝 Summary")
            st.info(analysis.summary)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("✅ Strengths")
                for strength in analysis.strengths:
                    st.write(f"- {strength}")
            with col4:
                st.subheader("⚠️ Weaknesses")
                for weakness in analysis.weaknesses:
                    st.write(f"- {weakness}")

        except Exception as e:
            st.error(f"❌ Execution Error: {str(e)}")