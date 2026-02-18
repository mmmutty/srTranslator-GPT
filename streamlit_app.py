import streamlit as st
import re
import time
import json
import requests
import pandas as pd  # ★表を作るために追加
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ Configuration & Constants
# ==========================================

CANDIDATE_MODELS = [
    "gpt-5.2",          # 【本命】推奨：賢くて4oより少し安い
    "gpt-5-mini",       # 【コスパ】テスト用
    "gpt-5.2-pro",      # 【超高性能】思考型
    "gpt-4o",           # 【安定】
    "gpt-5-nano"        # 【最安】
]

BATCH_SIZE = 20 

# ==========================================
# 🛠️ Helper Functions
# ==========================================

def search_movie_context(movie_title):
    query = f"{movie_title} movie script synopsis characters plot"
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return None
        combined_text = ""
        for res in results:
            try:
                page = requests.get(res['href'], timeout=3)
                if page.status_code == 200:
                    soup = BeautifulSoup(page.content, 'html.parser')
                    paragraphs = [p.get_text() for p in soup.find_all('p')]
                    combined_text += " ".join(paragraphs)[:3000]
            except: continue
        return combined_text if combined_text else None
    except: return None

def generate_style_guide(api_key, movie_title, raw_web_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    prompt = f"Read the info about '{movie_title}' and create a translation style guide."
    data = {
        "model": "gpt-5-mini", 
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": raw_web_text}]
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

def check_api(api_key):
    """API接続テスト"""
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        data = {
            "model": "gpt-5-mini", 
            "messages": [{"role":"user", "content":"hi"}], 
            "max_completion_tokens": 100 
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
        
        if res.status_code == 200:
            return True, "OK"
        else:
            try:
                err_msg = res.json().get('error', {}).get('message', res.text)
            except:
                err_msg = res.text
            return False, f"API Error ({res.status_code}): {err_msg}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def split_srt_blocks(srt_content):
    content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    return [b for b in re.split(r'\n\s*\n', content.strip()) if b.strip()]

def calculate_max_chars(timecode_line):
    """タイムコードから表示秒数を計算し、文字数上限を返す"""
    try:
        start_str, end_str = timecode_line.split('-->')
        def parse_seconds(t_str):
            h, m, s_ms = t_str.strip().split(':')
            s, ms = s_ms.replace(',', '.').split('.')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
            
        duration = parse_seconds(end_str) - parse_seconds(start_str)
        max_chars = max(5, min(int(duration * 4), 30))
        return max_chars
    except:
        return 20

def translate_batch(items, api_key, model_name, movie_title, target_lang, style_guide, previous_summary):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    input_dict = {
        str(i+1): {"text": item["text"], "max_chars_limit": item["max_chars"]} 
        for i, item in enumerate(items)
    }
    input_text = json.dumps(input_dict, ensure_ascii=False)
    
    context_str = ""
    if style_guide: context_str += f"[MOVIE INFO]\n{style_guide}\n"
    if previous_summary: context_str += f"[PREVIOUS CONTEXT]\n{previous_summary}\n"

    system_prompt = f"""
    You are a professional subtitle translator for "{movie_title}".
    Translate the provided JSON texts into natural {target_lang}.
    {context_str}
    
    CRITICAL RULES FOR SUBTITLES:
    1. Output MUST be a valid JSON object matching the input keys (IDs).
       Example Output format: {{"1": "こんにちは", "2": "元気？"}}
    2. STRICT LENGTH LIMIT PER ITEM: 
       You MUST strictly limit your translation length to the `max_chars_limit` specified for each ID.
       (e.g., If max_chars_limit is 8, your Japanese translation must be 8 characters or fewer.)
    3. NUANCE PRESERVATION:
       Do not translate word-for-word. Summarize to fit the limit, BUT absorb the nuance, emotion, and tone into natural Japanese sentence-ending particles (e.g., 〜よね, 〜さ, 〜だわ). Keep the character's soul alive.
    """

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4000,
        "temperature": 0.3 # ★AIの「ルール破り」を減らすために少し低めに設定
    }

    for _ in range(3):
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=150)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                parsed = json.loads(content)
                
                translated_lines = []
                for i in range(len(items)):
                    key = str(i + 1)
                    if key in parsed and parsed[key].strip():
                        translated_lines.append(parsed[key])
                    else:
                        translated_lines.append(items[i]["text"]) 
                return translated_lines
            elif res.status_code == 429:
                time.sleep(5)
                continue
        except Exception as e:
            time.sleep(2)
            
    return [item["text"] for item in items]

# ==========================================
# 🖥️ Main App
# ==========================================

def main():
    st.set_page_config(page_title="AI Subtitle Translator Pro", layout="wide")
    st.title("🎬 AI Subtitles Translator Pro")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", CANDIDATE_MODELS, index=0)
        st.markdown("---")
        title = st.text_input("Movie Title")
        lang = st.text_input("Target Language", value="Japanese")
        use_context = st.checkbox("Web Context Search", value=True)
        batch_size = st.slider("Batch Size", 10, 50, 20)

    uploaded_file = st.file_uploader("Upload SRT", type=["srt"])

    if uploaded_file and st.button("Start Translation", type="primary"):
        if not api_key:
            st.error("⚠️ API Key is missing.")
            return

        status = st.empty()
        p_bar = st.progress(0)
        log = st.empty()

        status.text("📡 Connecting to OpenAI...")
        is_connected, msg = check_api(api_key)

        if is_connected:
            status.success("✅ Connected!")
            
            # 1. Context Search
            style_guide = None
            if use_context and title:
                status.info("🌍 Searching context...")
                web_data = search_movie_context(title)
                if web_data:
                    style_guide = generate_style_guide(api_key, title, web_data)
                    st.expander("Generated Style Guide").markdown(style_guide)
                else:
                    st.warning("Web search found nothing, proceeding without context.")

            # 2. Prepare Blocks
            raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            blocks = split_srt_blocks(raw)
            total_blocks = len(blocks)
            
            parsed_blocks = []
            for b in blocks:
                lines = b.split('\n')
                if len(lines) >= 3:
                    t_idx = next((i for i, l in enumerate(lines) if '-->' in l), -1)
                    if t_idx != -1:
                        timecode = lines[t_idx]
                        max_c = calculate_max_chars(timecode) 
                        parsed_blocks.append({
                            "header": lines[:t_idx+1],
                            "text": "\n".join(lines[t_idx+1:]),
                            "original_block": b,
                            "max_chars": max_c 
                        })
                    else:
                        parsed_blocks.append({"original_block": b, "text": "", "max_chars": 0})
                else:
                    parsed_blocks.append({"original_block": b, "text": "", "max_chars": 0})

            translated_srt = []
            previous_context_summary = ""
            
            # ★文字数オーバーを記録するリスト
            overflow_reports = []
            
            status.info(f"🚀 Translating {total_blocks} lines using {model}...")

            for i in range(0, len(parsed_blocks), batch_size):
                batch = parsed_blocks[i : i + batch_size]
                items_to_translate = [{"text": b["text"], "max_chars": b["max_chars"]} for b in batch if b.get("text")]
                
                if items_to_translate:
                    translations = translate_batch(
                        items_to_translate, api_key, model, title, lang, style_guide, previous_context_summary
                    )
                    
                    trans_idx = 0
                    current_batch_text = ""
                    for b in batch:
                        if b.get("text"):
                            t_text = translations[trans_idx] if trans_idx < len(translations) else b["text"]
                            new_block = "\n".join(b["header"]) + "\n" + t_text + "\n\n"
                            translated_srt.append(new_block)
                            current_batch_text += t_text + " "
                            
                            # ★文字数チェック（改行コードを除外して純粋な文字数をカウント）
                            pure_text_len = len(t_text.replace('\n', '').replace('\r', ''))
                            if pure_text_len > b["max_chars"]:
                                overflow_reports.append({
                                    "No.": b["header"][0].strip(),
                                    "制限": b["max_chars"],
                                    "実際の文字数": pure_text_len,
                                    "オーバー量": f"+{pure_text_len - b['max_chars']}",
                                    "翻訳結果": t_text.replace('\n', ' ')
                                })
                            
                            trans_idx += 1
                        else:
                            translated_srt.append(b["original_block"] + "\n\n")
                    
                    previous_context_summary = current_batch_text[-200:]
                else:
                    for b in batch:
                        translated_srt.append(b["original_block"] + "\n\n")

                progress = min((i + batch_size) / total_blocks, 1.0)
                p_bar.progress(progress)
                log.text(f"Processing... {i}/{total_blocks}")

            p_bar.progress(1.0)
            status.success("✅ Translation Done!")
            
            # ダウンロードボタン
            st.download_button(
                "📥 Download Translated SRT", 
                "".join(translated_srt).encode('utf-8-sig'), 
                f"{uploaded_file.name}_AI.srt"
            )
            
            st.markdown("---")
            
            # ★文字数オーバーのレポート表示
            st.subheader("📊 翻訳クオリティ・レポート")
            if overflow_reports:
                st.warning(f"⚠️ {len(overflow_reports)} 個の字幕が「1秒4文字のルール（最大30文字）」をオーバーしています。手動での微調整をおすすめします。")
                df_report = pd.DataFrame(overflow_reports)
                # DataFrameを画面いっぱいにリッチに表示
                st.dataframe(df_report, use_container_width=True)
            else:
                st.success("✨ 素晴らしい！すべての字幕が文字数制限内に収まっています。")

        else:
            st.error(f"❌ Connection Failed.\nReason: {msg}")

if __name__ == "__main__":
    main()