import streamlit as st
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS  # 追加: 検索用

# ==========================================
# ⚙️ Configuration & Constants
# ==========================================

CANDIDATE_MODELS = [
    "gpt-4o-mini",      # 【おすすめ】コスパ最強
    "gpt-4o",           # 【最強】精度重視
    "gpt-4-turbo"
]

# ==========================================
# 🛠️ Helper Functions (Web Search & Context)
# ==========================================

def search_movie_context(movie_title):
    """
    映画タイトルから脚本やあらすじを検索し、テキスト情報を取得する
    """
    # 検索クエリ: タイトル + script/synopsis/transcript
    query = f"{movie_title} movie script transcript synopsis characters plot"
    
    try:
        # DuckDuckGoで検索 (上位3件)
        results = DDGS().text(query, max_results=3)
        if not results:
            return None
            
        combined_text = ""
        # 検索結果のURLからテキストを取得（簡易スクレイピング）
        for res in results:
            url = res['href']
            try:
                # タイムアウトを短めに設定して取得
                page = requests.get(url, timeout=3)
                if page.status_code == 200:
                    soup = BeautifulSoup(page.content, 'html.parser')
                    # <p>タグのテキストを集める（本文の可能性が高いため）
                    paragraphs = [p.get_text() for p in soup.find_all('p')]
                    # 最初の3000文字程度を取得（トークン節約）
                    text_content = " ".join(paragraphs)[:3000]
                    combined_text += f"\n--- Source: {url} ---\n{text_content}\n"
            except:
                continue
        
        return combined_text if combined_text else None
    except Exception as e:
        # エラー時はNoneを返して翻訳処理自体は止めない
        print(f"Search Error: {e}")
        return None

def generate_style_guide(api_key, movie_title, raw_web_text):
    """
    Webの情報を基に、翻訳用のスタイルガイド（設定資料）をAIに作成させる
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 指示書作成用のプロンプト
    system_prompt = f"""
    You are an expert movie localization director.
    Read the provided web content about the movie "{movie_title}".
    Create a concise "Translation Style Guide" for Japanese subtitles.
    
    Output Format:
    - **Genre & Tone**: (e.g., Serious, Slang-heavy, Historical, Comedy)
    - **Key Characters & Relationships**: (Who is talking to whom? e.g., "Jack and Rose are lovers", "Boss and subordinate")
    - **Speaking Style**: (e.g., "Use polite Desu/Masu", "Use rough Yakuza slang", "Old Samurai dialect")
    - **Plot Summary**: (Very brief summary to understand context)
    """

    data = {
        "model": "gpt-4o-mini", # 安価なモデルで十分
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Web Content:\n{raw_web_text}"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except:
        pass
    return None

# ==========================================
# 🛠️ Core Functions
# ==========================================

def find_working_model(api_key, log_area):
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    test_data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Test"}], "max_tokens": 5}
    log_area.text(f"👉 Testing API connection...")
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(test_data), timeout=10)
        if response.status_code == 200:
            log_area.success(f"✅ Connection successful!")
            return True
        else:
            st.warning(f"⚠️ Connection failed (Status: {response.status_code})")
            return False
    except Exception as e:
        st.error(f"📡 Connection Error: {str(e)}")
        return False

def split_srt_blocks(srt_content):
    content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    blocks = re.split(r'\n\s*\n', content.strip())
    return [b for b in blocks if b.strip()]

def sanitize_timecode(time_str):
    t = re.sub(r'\s*[-=]+>\s*', ' --> ', time_str)
    return t.replace('.', ',')

def translate_block_openai(text, api_key, model_name, movie_title, target_language, style_guide=None, previous_context=None):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    # 文脈情報の構築
    context_str = ""
    if style_guide:
        context_str += f"[MOVIE SETTING]\n{style_guide}\n\n"
    
    if previous_context:
        context_str += f"[PREVIOUS CONVERSATION]\n{previous_context}\n(Use this to understand the flow, but DO NOT translate these lines.)\n\n"

    system_prompt = f"""
    You are a professional subtitle translator for the movie "{movie_title}".
    Translate the [CURRENT LINE] into natural {target_language}.

    Guidelines:
    1. **Context Aware**: Look at [PREVIOUS CONVERSATION] to determine omitted subjects (who is "I", "You", "He"?) and the correct nuance.
       - Example: If previous line is "You are talented", "It's natural" -> "生まれつきさ" (Not "自然体").
    2. **Character Tone**: Reflect the character's personality defined in [MOVIE SETTING].
    3. **Format**: Output ONLY the translated text for [CURRENT LINE]. No quotes, no notes.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[CURRENT LINE]\n{text}"}
    ]
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3
    }

    # リトライ処理などは既存と同じ
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                return content if content else text
            elif response.status_code == 429:
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
    st.set_page_config(page_title="AI Subtitle Translator + Web Context", layout="wide")
    st.title("🎬 AI Subtitles Translator (ChatGPT)")

    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input("OpenAI API Key", type="password")
        selected_model = st.selectbox("Select Model", CANDIDATE_MODELS, index=0)
        st.markdown("---")
        
        # 映画タイトル入力（検索に必須）
        movie_title_input = st.text_input("Movie Title (Required for Context)", help="正確に入力すると検索精度が上がります")
        target_lang_input = st.text_input("Target Language", value="Japanese")
        
        # コンテキスト検索機能のON/OFF
        use_context = st.checkbox("🔍 Search Web for Context", value=True, help="ネットから脚本やあらすじを探して翻訳精度を上げます")

    uploaded_file = st.file_uploader("Upload SRT file", type=["srt"])

    if uploaded_file is not None and st.button("Start Translation", type="primary"):
        if not api_key_input:
            st.error("⚠️ API Key is missing.")
            return
        if use_context and not movie_title_input:
            st.warning("⚠️ To use Web Search, please enter the 'Movie Title'.")
            return

        status_area = st.empty()
        log_area = st.empty()
        context_expander = st.expander("📚 Generated Style Guide (Context)", expanded=False)
        progress_bar = st.progress(0)

        if find_working_model(api_key_input, log_area):
            
            # --- PHASE 1: Web Context Search & Analysis ---
            style_guide = None
            if use_context:
                status_area.info(f"🌍 Searching web for info about '{movie_title_input}'...")
                
                # 1. 検索 & スクレイピング
                raw_web_data = search_movie_context(movie_title_input)
                
                if raw_web_data:
                    status_area.info("📝 Generating style guide from web data...")
                    # 2. 情報を要約してスタイルガイド作成
                    style_guide = generate_style_guide(api_key_input, movie_title_input, raw_web_data)
                    
                    if style_guide:
                        context_expander.markdown(style_guide) # ユーザーに見えるように表示
                        st.toast("Style Guide Created Successfully!", icon="✅")
                    else:
                        st.warning("Could not generate style guide.")
                else:
                    st.warning("No relevant info found on the web. Proceeding without context.")
            
            # --- PHASE 2: Translation ---
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            blocks = split_srt_blocks(content)
            total_blocks = len(blocks)
            translated_srt = []
            
            # ★追加: 直前の会話を保存するリスト（バッファ）
            conversation_history = [] 
            
            status_area.info(f"🚀 Translating {total_blocks} blocks with Context Flow...")
            
            for i, block in enumerate(blocks):
                lines = block.split('\n')
                if len(lines) >= 2:
                    time_line_index = -1
                    for idx, line in enumerate(lines):
                        if '-->' in line:
                            time_line_index = idx
                            break
                    
                    if time_line_index != -1:
                        seq_num = lines[0]
                        timecode = lines[time_line_index]
                        original_text = "\n".join(lines[time_line_index + 1:])
                        
                        if original_text.strip():
                            # ★変更: 直近3件の履歴をテキスト化して渡す
                            previous_context_str = "\n".join(conversation_history[-3:]) # 直前3ブロック分
                            
                            translated_text = translate_block_openai(
                                original_text, 
                                api_key_input, 
                                selected_model, 
                                movie_title_input, 
                                target_lang_input,
                                style_guide=style_guide,
                                previous_context=previous_context_str # ★ここで過去の文脈を渡す
                            )
                            
                            # ★追加: 翻訳に使った原文を履歴に追加
                            # (改行を除去して1行にして保存すると読みやすい)
                            clean_original = original_text.replace('\n', ' ')
                            conversation_history.append(clean_original)
                            
                        else:
                            translated_text = ""
                        
                        clean_time = sanitize_timecode(timecode)
                        new_block = f"{seq_num}\r\n{clean_time}\r\n{translated_text}\r\n\r\n"
                        translated_srt.append(new_block)
                    else:
                        translated_srt.append(block + "\r\n\r\n")
                else:
                    translated_srt.append(block + "\r\n\r\n")
                
                # --- 以下、進捗バーなどの既存コード ---
                progress = (i + 1) / total_blocks
                progress_bar.progress(progress)
                if (i + 1) % 5 == 0:
                    log_area.text(f"⏳ Processing... {i + 1}/{total_blocks}")
                time.sleep(0.05)

            progress_bar.progress(1.0)
            status_area.success("✅ Complete!")
            
            # ダウンロードボタン
            final_content = "".join(translated_srt)
            new_filename = f"{uploaded_file.name.replace('.srt', '')}_AI_WebContext.srt"
            
            st.download_button(
                label="📥 Download Translated SRT",
                data=final_content.encode('utf-8-sig'),
                file_name=new_filename,
                mime="text/plain"
            )

if __name__ == "__main__":
    main()