import streamlit as st
import re
import time
import json
import requests
import os

# ==========================================
# ⚙️ Configuration & Constants
# ==========================================

# Latest OpenAI Model List (Updated for 2026)
CANDIDATE_MODELS = [
    "gpt-4o-mini",      # 【おすすめ】最新の軽量モデル（爆速・激安・高性能）
    "gpt-4o",           # 【最強】現在のフラッグシップモデル（賢いが価格はminiの約30倍）
    "o1-mini",          # 【推理】思考型モデルの軽量版（字幕には少し遅いかも）
    "gpt-4-turbo"       # 一つ前の高性能モデル
]

# ==========================================
# 🛠️ Function Definitions
# ==========================================

def find_working_model(api_key, log_area):
    """Function to check OpenAI API connection"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 接続テストは一番安いモデルで行う
    test_data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 5
    }

    log_area.text(f"👉 Testing API connection...")
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            data=json.dumps(test_data), 
            timeout=10
        )
        
        if response.status_code == 200:
            log_area.success(f"✅ Connection successful! Ready to use OpenAI.")
            return True
        else:
            try:
                error_msg = response.json().get('error', {}).get('message', response.text)
            except:
                error_msg = response.text
            st.warning(f"⚠️ Connection failed (Status: {response.status_code})\nReason: {error_msg}")
            return False
            
    except Exception as e:
        st.error(f"📡 Connection Error: {str(e)}")
        return False

def split_srt_blocks(srt_content):
    # Enhanced logic to prevent syncing issues
    content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', content.strip())
    return [b for b in blocks if b.strip()]

def sanitize_timecode(time_str):
    """Strictly format timecode for Web tools"""
    t = re.sub(r'\s*[-=]+>\s*', ' --> ', time_str)
    t = t.replace('.', ',')
    return t

def translate_block_openai(text, api_key, model_name, movie_title, target_language):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # プロンプトの定義
    system_prompt = f"""
    You are a professional film subtitle translator.
    Translate the dialogue into natural, emotional {target_language}.
    Movie: {movie_title}
    
    Rules:
    1. Output ONLY the translated text. No notes.
    2. Do NOT output timecodes.
    3. Keep it concise for subtitles.
    """
    
    # モデルごとの仕様対応（o1系はtemperatureなどが使えない場合があるため調整）
    if model_name.startswith("o1"):
        # o1モデルは "system" ロールが推奨されない場合があるため "user" に統合するか、
        # "developer" ロールを使うが、簡易的にuserで処理
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nOriginal Text to Translate:\n{text}"}
        ]
        data = {
            "model": model_name,
            "messages": messages,
            # o1系は max_completion_tokens を使うが、汎用性のためパラメータを最小限に
        }
    else:
        # GPT-4o系
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.3
        }

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60) # o1は遅いのでタイムアウト長め
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                return content if content else text
            elif response.status_code == 429:
                # Rate limit
                time.sleep(5)
                continue
            else:
                time.sleep(1)
                continue
        except:
            time.sleep(1)
            continue
            
    return text

# ==========================================
# 🖥️ Streamlit Screen Layout
# ==========================================

def main():
    st.set_page_config(page_title="AI Subtitle Translator (OpenAI)", layout="wide")
    
    st.title("🎬 AI Subtitles Translator (ChatGPT)")

    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        st.markdown("---")
        
        # Model Selection
        selected_model = st.selectbox("Select Model", CANDIDATE_MODELS, index=0)
        
        # モデルの説明表示
        if selected_model == "gpt-4o-mini":
            st.success("✅ Recommended! Fastest & Cheapest.")
        elif selected_model == "gpt-4o":
            st.warning("💰 High Cost. Highest Quality.")
        elif selected_model.startswith("o1"):
            st.info("🧠 Reasoning Model. Slower but deeper understanding.")

        st.markdown("---")
        movie_title_input = st.text_input("Movie Title")
        target_lang_input = st.text_input("Target Language", value="Japanese")
        st.markdown("---")
        st.info("Ensure you have credit balance in OpenAI platform.")

    uploaded_file = st.file_uploader("Drag and drop your SRT file here", type=["srt"])

    if uploaded_file is not None:
        st.success(f"File loaded: {uploaded_file.name}")
        
        if st.button("Start Translation", type="primary"):
            if not api_key_input:
                st.error("⚠️ Please enter your API Key in the sidebar.")
                return

            status_area = st.empty()
            log_area = st.empty()
            progress_bar = st.progress(0)

            # Check connection
            if find_working_model(api_key_input, log_area):
                content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                blocks = split_srt_blocks(content)
                total_blocks = len(blocks)
                translated_srt = []
                
                status_area.info(f"🚀 Starting translation... Total {total_blocks} blocks (Model: {selected_model})")
                
                for i, block in enumerate(blocks):
                    lines = block.split('\n')
                    if len(lines) >= 2:
                        seq_num = lines[0].strip()
                        
                        time_line_index = -1
                        for idx, line in enumerate(lines):
                            if '-->' in line:
                                time_line_index = idx
                                break
                        
                        if time_line_index != -1:
                            timecode = lines[time_line_index].strip()
                            original_text = "\n".join(lines[time_line_index + 1:])
                            
                            if original_text.strip():
                                translated_text = translate_block_openai(
                                    original_text, 
                                    api_key_input, 
                                    selected_model, 
                                    movie_title_input, 
                                    target_lang_input
                                )
                            else:
                                translated_text = ""
                            
                            clean_time = sanitize_timecode(timecode)
                            new_block = f"{seq_num}\r\n{clean_time}\r\n{translated_text}\r\n\r\n"
                            translated_srt.append(new_block)
                        else:
                            translated_srt.append(block.replace('\n', '\r\n') + "\r\n\r\n")
                    else:
                        translated_srt.append(block.replace('\n', '\r\n') + "\r\n\r\n")
                    
                    progress = (i + 1) / total_blocks
                    progress_bar.progress(progress)
                    
                    if (i + 1) % 10 == 0:
                         log_area.text(f"⏳ Processing... {i + 1}/{total_blocks} completed")
                    
                    # 4o-miniは非常に高速ですが、連続リクエスト制限(Rate Limit)を避けるため少し待機
                    time.sleep(0.1)

                progress_bar.progress(1.0)
                status_area.success("✅ Translation & Formatting Complete!")
                log_area.empty()
                
                final_content = "".join(translated_srt)
                new_filename = f"{uploaded_file.name.replace('.srt', '')}_{target_lang_input}_{selected_model}_WebReady.srt"
                
                st.download_button(
                    label="📥 Download Translated SRT",
                    data=final_content.encode('utf-8-sig'),
                    file_name=new_filename,
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()