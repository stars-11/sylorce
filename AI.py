from typing import List
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser 
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_models import ChatOpenAI
import time

st.set_page_config(page_title="Sylorce", layout="centered")
st.title("Sylorce")

@st.cache_resource
def load_model():
    return ChatOpenAI(
        api_key="sk-9c0aa717ff8248d5a66bef5488239d3d",  
        base_url="https://api.deepseek.com/v1", 
        model="deepseek-chat",  
        temperature=0.7  
    )

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_response" not in st.session_state:
    st.session_state.current_response = ""

def to_message_place_holder(messages: List[dict]) -> List:
    return [
        AIMessage(content=message['content']) if message['role'] == "assistant" 
        else HumanMessage(content=message['content'])
        for message in messages
    ]

memory_key = "history"
prompt_template = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name=memory_key),
    ('human', '{input}')
])

def create_chain():
    return {
        'input': lambda x: x['input'],
        'history': lambda x: to_message_place_holder(x['messages'])
    } | prompt_template | model | StrOutputParser()

def generate_response(user_input):
    st.session_state.processing = True
    st.session_state.current_response = ""
    
    chain = create_chain()
    full_response = ""
    
    try:
        for chunk in chain.stream({
            'input': user_input,
            'messages': st.session_state.messages
        }):
            full_response += chunk
            st.session_state.current_response = full_response
            time.sleep(0.01)
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })
        
    except Exception as e:
        st.error(f"生成响应时出错: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "抱歉，生成响应时出现了问题。请稍后再试。"
        })
    
    finally:
        st.session_state.processing = False
        st.session_state.current_response = ""


for i, message in enumerate(st.session_state.messages):

    if message["role"] == "user":
        st.markdown(f"**您:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")
    st.divider()

if st.session_state.processing and st.session_state.current_response:
    st.markdown(f"**AI:** {st.session_state.current_response}")
    st.divider()

user_input = st.chat_input(
    "请输入您的问题", 
    key=f"chat_input_{len(st.session_state.messages)}",
    disabled=st.session_state.processing
)

if user_input and not st.session_state.processing:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    st.rerun()

if (st.session_state.messages and 
    st.session_state.messages[-1]["role"] == "user" and 
    not st.session_state.processing):
    
    generate_response(st.session_state.messages[-1]["content"])
    st.rerun()

if st.button("清空聊天记录", key="clear_chat"):
    st.session_state.messages = []
    st.session_state.current_response = ""
    st.rerun()

st.markdown("""
<style>
    .stMarkdown {
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)