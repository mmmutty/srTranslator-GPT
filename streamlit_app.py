import streamlit as st
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# ==========================================
# ⚙️ Configuration & Constants
# ==========================================

# 2026年最新モデル構成
CANDIDATE_MODELS = [
    "gpt-5.2",          # 【最新】推奨：賢くて4oより少し安い
    "gpt-5-mini",       # 【コスパ】テスト用
    "gpt-5-nano",       # 【爆速】
    "gpt-4o"            # 【安定】
]

# 一度に翻訳する字幕の行数（多すぎるとAIが混乱し、少なすぎると遅い）
BATCH_SIZE = 20 

# ==========================================
# 🛠️ Helper Functions
# ==========================================

def search_movie_context(movie_title):
    """映画の情報を検索して取得"""
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
    """検索結果からスタイルガイドを生成"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    prompt = f"""
    Read the info about "{movie_title}" and create a translation style guide.
    Output: Genre/Tone, Character Relationships, Speaking Styles (polite/slang), Plot Summary.
    """
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
        data = {"model": "gpt-5-mini", "messages": [{"role":"user", "content":"hi"}], "max_tokens":1}
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=5)
        return res.status_code == 200
    except: return False

def split_srt_blocks(srt_content):
    content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
    return [b for b in re.split(r'\n\s*\n', content.strip()) if b.strip()]

def sanitize_timecode(time_str):
    return re.sub(r'\s*[-=]+>\s*', ' --> ', time_str).replace('.', ',')

# ==========================================
# 🚀 Batch Translation Function
# ==========================================

def translate_batch(lines, api_key, model_name, movie_title, target_lang, style_guide, previous_summary):
    """
    複数のセリフ(lines)をまとめて翻訳する関数
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    # 入力テキストを行番号付きで整形
    input_text = "\n".join([f"[{i+1}] {line}" for i, line in enumerate(lines)])
    
    context_str = ""
    if style_guide: context_str += f"[MOVIE INFO]\n{style_guide}\n"
    if previous_summary: context_str += f"[PREVIOUS CONTEXT]\n{previous_summary}\n"

    system_prompt = f"""
    You are a professional subtitle translator for "{movie_title}".
    Translate the following {len(lines)} lines into natural {target_lang}.

    {context_str}

    Rules:
    1. Maintain the context flow between lines.
    2. Respect the character tones from Movie Info.
    3. Output format must be a JSON list of strings strictly matching the input order.
    Example Input:
    [1] Hello.
    [2] How are you?
    Example Output:
    ["こんにちは。", "元気？"]
    """

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ],
        "response_format": {"type": "json_object"} # JSONモードを強制（GPT-4o/5系で有効）
    }

    for _ in range(3): # リトライ3回
        try:
            res = requests.post(url, headers=headers, data=json.dumps(data), timeout=120)
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                # JSONをパースしてリストを取り出す
                parsed = json.loads(content)
                # キーが "translations" だったりリスト直下だったりする場合の揺らぎ吸収
                if isinstance(parsed, dict):
                    # 辞書内の最初のリスト要素を取得
                    values = list(parsed.values())
                    if values and isinstance(values[0], list):
                        return values[0]
                elif isinstance(parsed, list):
                    return parsed
                
                # 失敗時は原文を返す（エラー回避）
                return lines
            elif res.status_code == 429:
                time.sleep(5)
                continue
        except Exception as e:
            print(e)
            time.sleep(1)
            
    return lines # 全失敗時は原文を返す

# ==========================================
# 🖥️ Main App
# ==========================================

def main():
    st.set_page_config(page_title="AI Subtitle Translator Pro", layout="wide")
    st.title("🎬 AI Subtitles Translator Pro (Batch Mode)")

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", CANDIDATE_MODELS)
        st.markdown("---")
        title = st.text_input("Movie Title", help="Context search key")
        lang = st.text_input("Target Language", value="Japanese")
        use_context = st.checkbox("Web Context Search", value=True)
        batch_size = st.slider("Batch Size", 10, 50, 20, help="一度に翻訳する行数。大きいほど速いがエラーが出やすい。")

    uploaded_file = st.file_uploader("Upload SRT", type=["srt"])

    if uploaded_file and st.button("Start Translation", type="primary"):
        if not api_key:
            st.error("API Key missing.")
            return

        status = st.empty()
        p_bar = st.progress(0)
        log = st.empty()

        if check_api(api_key):
            # 1. Context Search
            style_guide = None
            if use_context and title:
                status.info("🌍 Searching context...")
                web_data = search_movie_context(title)
                if web_data:
                    style_guide = generate_style_guide(api_key, title, web_data)
                    st.expander("Style Guide").markdown(style_guide)

            # 2. Prepare Blocks
            raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            blocks = split_srt_blocks(raw)
            total_blocks = len(blocks)
            
            # データを解析してリスト化 (ID, Time, Text)
            parsed_blocks = []
            for b in blocks:
                lines = b.split('\n')
                if len(lines) >= 3: # タイムコードとテキストがある場合
                     # タイムコード行を探す
                    t_idx = next((i for i, l in enumerate(lines) if '-->' in l), -1)
                    if t_idx != -1:
                        parsed_blocks.append({
                            "header": lines[:t_idx+1], # IDと時間
                            "text": "\n".join(lines[t_idx+1:]), # 字幕本文
                            "original_block": b
                        })
                    else:
                        parsed_blocks.append({"original_block": b, "text": ""})
                else:
                    parsed_blocks.append({"original_block": b, "text": ""})

            translated_srt = []
            
            # 3. Batch Loop
            status.info(f"🚀 Translating {total_blocks} lines in batches of {batch_size}...")
            
            # 直前の文脈（バッチ間のつなぎ用）
            previous_context_summary = ""

            for i in range(0, len(parsed_blocks), batch_size):
                batch = parsed_blocks[i : i + batch_size]
                
                # 翻訳が必要なテキストだけ抽出
                texts_to_translate = [b["text"] for b in batch if b.get("text")]
                
                if texts_to_translate:
                    # ★翻訳実行
                    translations = translate_batch(
                        texts_to_translate, api_key, model, title, lang, style_guide, previous_context_summary
                    )
                    
                    # 結果を割り当て & 次のコンテキスト用に保存
                    trans_idx = 0
                    current_batch_text = ""
                    
                    for b in batch:
                        if b.get("text"):
                            # 翻訳結果があればそれを使う、なければ原文
                            t_text = translations[trans_idx] if trans_idx < len(translations) else b["text"]
                            
                            # SRT再構築
                            new_block = "\n".join(b["header"]) + "\n" + t_text + "\n\n"
                            translated_srt.append(new_block)
                            
                            current_batch_text += t_text + " "
                            trans_idx += 1
                        else:
                            translated_srt.append(b["original_block"] + "\n\n")
                    
                    # 次のバッチのために、今回の終わりの方を記憶させておく
                    previous_context_summary = current_batch_text[-200:] # 後ろ200文字程度
                    
                else:
                    # 翻訳するテキストがないブロック（音楽など）
                    for b in batch:
                        translated_srt.append(b["original_block"] + "\n\n")

                # Progress
                progress = min((i + batch_size) / total_blocks, 1.0)
                p_bar.progress(progress)
                log.text(f"Processing... {i}/{total_blocks}")

            p_bar.progress(1.0)
            status.success("Done!")
            
            st.download_button(
                "📥 Download SRT", 
                "".join(translated_srt).encode('utf-8-sig'), 
                f"{uploaded_file.name}_AI.srt"
            )

if __name__ == "__main__":
    main()