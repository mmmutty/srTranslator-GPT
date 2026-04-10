import streamlit as st
import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ Configuration & Constants
# ==========================================

CANDIDATE_MODELS = [
    "gpt-5.2",          
    "gpt-5-mini",       
    "gpt-5.2-pro",      
    "gpt-4o",           
    "gpt-5-nano"        
]

BATCH_SIZE = 20 

# ==========================================
# 🛠️ Helper Functions
# ==========================================

def search_movie_context(movie_title):
    """検索クエリを最適化し、ヒット率を向上"""
    query = f"movie '{movie_title}' plot summary tone"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
            
        if not results:
            return None
            
        combined_text = ""
        for res in results:
            try:
                page = requests.get(res['href'], timeout=5)
                if page.status_code == 200:
                    soup = BeautifulSoup(page.content, 'html.parser')
                    for s in soup(['script', 'style']): s.decompose()
                    paragraphs = [p.get_text() for p in soup.find_all('p')]
                    combined_text += " ".join(paragraphs)[:2000]
            except: continue
            
        return combined_text if combined_text else None
    except Exception:
        return None

def generate_style_guide(api_key, movie_title, raw_text):
    """ハルシネーションを防ぐため、ジャンルとトーンだけを抽出"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    prompt = f"""
    Read the provided information about the movie "{movie_title}".
    Create a very brief translation style guide focusing ONLY on:
    1. Genre & Overall Tone (e.g., Sci-fi thriller, dark and serious)
    2. General Speaking Styles (e.g., natural, polite, or rough)
    
    Keep it concise. Do not guess character relationships if not explicitly stated.
    """
    
    data = {
        "model": "gpt-5-mini", 
        "messages": [
            {"role": "system", "content": prompt}, 
            {"role": "user", "content": raw_text}
        ],
        "temperature": 0.0 # ★想像を排除し、事実のみにフォーカス
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(data), timeout=20)
        return res.json()['choices'][0]['message']['content'] if res.status_code == 200 else None
    except: return None

def check_api(api_key):
    """API接続テスト（1単語返答で思考トークン上限エラーを回避）"""
    try:
        headers = {'Authorization': f'Bearer {api_key}'}
        data = {
            "model": "gpt-5-mini", 
            "messages": [
                {"role": "system", "content": "Reply with exactly one word."},
                {"role": "user", "content": "hi"}
            ]
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
        if res.status_code == 200:
            return True, "OK"
        else:
            err_msg = res.json().get('error', {}).get('message', res.text) if hasattr(res, 'json') else res.text
            return False, f"API Error ({res.status_code}): {err_msg}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def split_srt_blocks(srt_content):
    content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    return [b for b in re.split(r'\n\s*\n', content.strip()) if b.strip()]

def calculate_max_chars(timecode_line, target_lang):
    """言語に応じて1秒あたりの文字数上限を動的に計算する"""
    try:
        start_str, end_str = timecode_line.split('-->')
        def parse_seconds(t_str):
            h, m, s_ms = t_str.strip().split(':')
            s, ms = s_ms.replace(',', '.').split('.')
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
            
        duration = parse_seconds(end_str) - parse_seconds(start_str)
        
        cjk_keywords = ['japanese', 'korean', 'chinese', '日本語', '한국어', '中文', '台湾華語']
        is_cjk = any(keyword in target_lang.lower() for keyword in cjk_keywords)
        
        if is_cjk:
            max_chars = max(5, min(int(duration * 4.5), 30))
        else:
            max_chars = max(15, min(int(duration * 15), 80))
            
        return max_chars
    except:
        return 30 

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
       Example: {{"1": "Translated text 1", "2": "Translated text 2"}}
       
    2. LENGTH LIMIT vs NATURALNESS (The Golden Balance):
       - You must aim to stay within the `max_chars_limit`.
       - FATAL ERROR: DO NOT use robotic, fragmented, or telegraphic language (e.g., dropping essential grammatical particles).
       - To save space, DO NOT just delete words. Instead, PARAPHRASE into concise, natural conversational expressions.
       - If you must choose between "sounding like a robot" and "exceeding the limit by a few characters", choose exceeding the limit slightly. But try your absolute best to paraphrase nicely within the limit.
       
    3. NUANCE PRESERVATION:
       Absorb the emotion and tone into natural phrasing typical for native speakers of {target_lang}. Keep the dialogue completely natural and human-like.
    """

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4000,
        "temperature": 0.3
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
            
            style_guide = None
            if use_context and title:
                status.info("🌍 Searching context...")
                web_data = search_movie_context(title)
                if web_data:
                    style_guide = generate_style_guide(api_key, title, web_data)
                    st.expander("Generated Style Guide").markdown(style_guide)
                else:
                    st.warning("Web search found nothing, proceeding without context.")

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
                        max_c = calculate_max_chars(timecode, lang) 
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
            overflow_reports = []
            
            status.info(f"🚀 Translating {total_blocks} lines into {lang} using {model}...")

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
            
            st.download_button(
                "📥 Download Translated SRT", 
                "".join(translated_srt).encode('utf-8-sig'), 
                f"{uploaded_file.name}_AI.srt"
            )
            
            st.markdown("---")
            
            st.subheader("📊 翻訳クオリティ・レポート")
            if overflow_reports:
                st.warning(f"⚠️ {len(overflow_reports)} 個の字幕が「{lang}の文字数制限」をオーバーしています。")
                df_report = pd.DataFrame(overflow_reports)
                st.dataframe(df_report, use_container_width=True)
            else:
                st.success("✨ 素晴らしい！すべての字幕が文字数制限内に収まっています。")

        else:
            st.error(f"❌ Connection Failed.\nReason: {msg}")

if __name__ == "__main__":
    main()