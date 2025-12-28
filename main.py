import streamlit as st
import langchain
import langchain_huggingface
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import langchain_core
from langchain_core.prompts import PromptTemplate
import dotenv
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("hf")

st.title("AI Chatbot Mentor")

st.write("Welcome to AI Chatbot Mentor")
st.write("Your personalised AI learning assistant")
st.write("Select a module to start your mentoring journey")

domain = st.radio(
    "Available Learning Modules",
    (
        "Python",
        "SQL",
        "Power BI",
        "Exploratory Data Analysis (EDA)",
        "Machine Learning (ML)",
        "Deep Learning (DL)",
        "Generative AI",
        "Agentic AI"
    )
)

model = HuggingFaceEndpoint(repo_id = "deepseek-ai/DeepSeek-V3.2",temperature = 0.2)
deepseek = ChatHuggingFace(llm = model)

system_prompt_template = PromptTemplate(template="""You are an expert AI mentor strictly for the domain: {domain}.
                                    Rules you MUST follow:
                                    1. Answer ONLY questions related to {domain}.
                                    2. If the question does NOT belong to {domain}, DO NOT answer it.
                                    3. Instead, reply exactly with:
                                    Sorry, I don't know about this question. Please ask something related to the selected module.
                                    4. Do not explain why.
                                    5. Be clear, professional, and supportive.""",
                                    input_variables=["domain"])

def llm():
    if "current_topic" not in st.session_state:
        st.session_state["current_topic"] = domain
        st.session_state["conver"] = []
        st.session_state["memory"] = []

        system_prompt = system_prompt_template.format(domain = domain)
        st.session_state["memory"].append(("system",system_prompt))


    if st.session_state["current_topic"] != domain:
        st.session_state["current_topic"] = domain
        st.session_state["conver"] = []
        st.session_state["memory"] = []
        
        system_prompt = system_prompt_template.format(domain = domain)
        st.session_state["memory"].append(("system",system_prompt))

    user_message = st.chat_input("Ask your doubt")

    if user_message:
        st.session_state["memory"].append(("human",user_message))

        with st.spinner("Mentor is thinking..."):
            output = deepseek.invoke(st.session_state["memory"])

        st.session_state["memory"].append(("ai",output.content))
        st.session_state["conver"].append({"role":"human","data":user_message})
        st.session_state["conver"].append({"role":"ai","data":output.content})

if domain == "Python":
    st.write("Welcome to the Python AI Mentor")
    st.write("I am your domain-specific mentor for Python.")
    st.write("Ask me anything related to Python programming.")
    llm()

elif domain == "SQL":
    st.write("Welcome to the SQL AI Mentor")
    st.write("I am your domain-specific mentor for SQL.")
    st.write("Ask me anything related to SQL.")
    llm()

elif domain == "Power BI":
    st.write("Welcome to the Power BI AI Mentor")
    st.write("I am your domain-specific mentor for Power BI.")
    st.write("Ask me anything related to Power BI.")
    llm()

elif domain == "Exploratory Data Analysis(EDA)":
    st.write("Welcome to the Exploratory Data Analysis(EDA) AI Mentor")
    st.write("I am your domain-specific mentor for Exploratory Data Analysis(EDA).")
    st.write("Ask me anything related to Exploratory Data Analysis(EDA).")
    llm()

elif domain == "Machine Learning(ML)":
    st.write("Welcome to the Machine Learning(ML) AI Mentor")
    st.write("I am your domain-specific mentor for Machine Learning(ML).")
    st.write("Ask me anything related to Machine Learning(ML).")
    llm()

elif domain == "Deep Learning(ML)":
    st.write("Welcome to the Deep Learning(ML) AI Mentor")
    st.write("I am your domain-specific mentor for Deep Learning(ML).")
    st.write("Ask me anything related to Deep Learning(ML).")
    llm()

elif domain == "Generative AI":
    st.write("Welcome to the Generative AI AI Mentor")
    st.write("I am your domain-specific mentor for Generative AI.")
    st.write("Ask me anything related to Generative AI.")
    llm()

else:
    st.write("Welcome to the Agentic AI AI Mentor")
    st.write("I am your domain-specific mentor for Agentic AI.")
    st.write("Ask me anything related to Agentic AI.")
    llm()


if "conver" in st.session_state:
    for y in st.session_state["conver"]:
        with st.chat_message(y["role"]):
            st.write(y["data"])

if st.session_state.get("conver"):
    session = "\n\n".join([f"{msg['role'].upper()}: {msg['data']}" for msg in st.session_state["conver"]])

    with open("ai_chatbot_responses.txt", "w", encoding="utf-8") as file:
        file.write(session)

    download_btn = st.download_button(
        label= "Download Chat Session",
        data=session,
        file_name="ai_chatbot_responses.txt",
        mime="text/plain"
    )
    
    if download_btn:
        st.write("Downloaded entire successfully")