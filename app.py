import streamlit as st
import requests
import json
import time
from typing import List, Dict, Generator

# 设置页面配置
st.set_page_config(page_title="Sylorce", layout="centered")
st.title("Sylorce AI")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-9c0aa717ff8248d5a66bef5488239d3d"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def format_messages(self, history_messages: List[Dict]) -> List[Dict]:
        """将消息历史格式化为 API 要求的格式"""
        formatted_messages = []
        for msg in history_messages:
            if msg["role"] == "user":
                formatted_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                formatted_messages.append({"role": "assistant", "content": msg["content"]})
        return formatted_messages
    
    def stream_response(self, user_input: str, history_messages: List[Dict]) -> Generator[str, None, None]:
        """流式获取 AI 响应"""
        # 准备消息历史
        all_messages = self.format_messages(history_messages)
        all_messages.append({"role": "user", "content": user_input})
        
        # 准备请求数据
        payload = {
            "model": "deepseek-chat",
            "messages": all_messages,
            "temperature": 0.7,
            "stream": True,
            "max_tokens": 2000
        }
        
        try:
            # 发送流式请求
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = f"API 请求失败: {response.status_code} - {response.text}"
                st.error(error_msg)
                yield error_msg
                return
            
            # 处理流式响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            yield full_response
            
        except requests.exceptions.RequestException as e:
            error_msg = f"网络请求错误: {str(e)}"
            st.error(error_msg)
            yield error_msg
        except Exception as e:
            error_msg = f"处理响应时出错: {str(e)}"
            st.error(error_msg)
            yield error_msg

# 初始化 DeepSeek 客户端
@st.cache_resource
def get_deepseek_client():
    return DeepSeekClient(DEEPSEEK_API_KEY, DEEPSEEK_API_URL)

deepseek_client = get_deepseek_client()

# 处理 AI 响应
def generate_response(user_input: str):
    """生成 AI 响应并更新状态"""
    st.session_state.processing = True
    
    try:
        # 创建响应占位符
        response_placeholder = st.empty()
        full_response = ""
        
        # 流式获取响应
        for chunk in deepseek_client.stream_response(user_input, st.session_state.messages):
            if chunk:
                full_response += chunk
                response_placeholder.markdown(f"**AI:** {full_response}")
                time.sleep(0.01)  # 稍微延迟以改善流式效果
        
        # 添加助手消息到历史
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response
        })
        
    except Exception as e:
        error_msg = f"生成响应时出错: {str(e)}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "抱歉，生成响应时出现了问题。请稍后再试。"
        })
    
    finally:
        st.session_state.processing = False
        st.rerun()

# 显示聊天界面
def display_chat_interface():
    """显示聊天界面"""
    
    # 显示历史消息
    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f"**您:** {message['content']}")
        else:
            st.markdown(f"**AI:** {message['content']}")
        st.divider()
    
    # 用户输入
    user_input = st.chat_input(
        "请输入您的问题...", 
        key=f"chat_input_{len(st.session_state.messages)}",
        disabled=st.session_state.processing
    )
    
    # 处理用户输入
    if user_input and not st.session_state.processing:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 立即显示用户消息
        st.markdown(f"**您:** {user_input}")
        st.divider()
        
        # 生成 AI 响应
        generate_response(user_input)

# 侧边栏功能
def sidebar_controls():
    """侧边栏控制"""
    with st.sidebar:
        st.header("控制面板")
        
        # 显示当前消息数量
        st.info(f"当前对话: {len(st.session_state.messages)} 条消息")
        
        # 清空聊天记录按钮
        if st.button("🗑️ 清空聊天记录", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        
        
# 主应用
def main():
    """主应用函数"""
    # 显示侧边栏
    sidebar_controls()
    
    # 显示聊天界面
    display_chat_interface()
    
    # 添加自定义 CSS
    st.markdown("""
    <style>
    .stMarkdown {
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 70%;
        left: 15%;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
