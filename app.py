from __future__ import annotations

import base64
import io
import zipfile
from datetime import datetime
from uuid import uuid4

import streamlit as st
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)


st.set_page_config(
    page_title="画像生成スタジオ",
    page_icon="🎨",
    layout="wide",
)

MODEL_OPTIONS = {
    "高精度な編集向け（Sunburst）": "gpt-image-2.5-sunburst",
    "スピード重視（Flare）": "gpt-image-2.5-flare",
}

SIZE_OPTIONS = {
    "正方形 1:1（1024 × 1024）": "1024x1024",
    "縦長 9:16（1008 × 1792）": "1008x1792",
    "縦長 2:3（1024 × 1536）": "1024x1536",
    "横長 16:9（1792 × 1008）": "1792x1008",
    "横長 3:2（1536 × 1024）": "1536x1024",
}

QUALITY_OPTIONS = {
    "低（low）— ラフ案向け": "low",
    "標準（medium）": "medium",
    "高（high）": "high",
    "とても高い（xhigh）": "xhigh",
    "最大（max）— 時間・費用が増える場合あり": "max",
    "自動（auto）": "auto",
}

if "history" not in st.session_state:
    st.session_state.history = []


def get_secret_api_key() -> str:
    """Read API key from Streamlit secrets if configured."""
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        # No secrets.toml is also a normal setup; user can enter the key in the sidebar.
        return ""


def make_api_image(filename: str, image_bytes: bytes, mime_type: str):
    """Return a file tuple accepted by the OpenAI Python SDK."""
    return (filename, io.BytesIO(image_bytes), mime_type)


def make_zip(history_items: list[dict]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, item in enumerate(history_items, start=1):
            filename = f"image_{index:02d}_{item['id']}.png"
            archive.writestr(filename, item["image_bytes"])
    return zip_buffer.getvalue()


def add_results(result, prompt: str, model: str, size: str, quality: str, mode: str) -> int:
    """Add returned PNG images to this browser session's in-memory history."""
    new_items = []
    for image_data in result.data or []:
        if not getattr(image_data, "b64_json", None):
            continue
        raw = base64.b64decode(image_data.b64_json)
        new_items.append(
            {
                "id": uuid4().hex[:8],
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": prompt,
                "model": model,
                "size": size,
                "quality": quality,
                "mode": mode,
                "image_bytes": raw,
            }
        )
    # Most recent first. Results are kept only in Streamlit session memory.
    st.session_state.history = new_items + st.session_state.history
    return len(new_items)


st.title("🎨 画像生成スタジオ")
st.caption("OpenAI APIを使って、テキストからの画像生成と参考画像の編集を行います。生成した画像はPNGで保存できます。")

with st.sidebar:
    st.header("API設定")
    default_key = get_secret_api_key()
    api_key = st.text_input(
        "OpenAI APIキー",
        value=default_key,
        type="password",
        help=".streamlit/secrets.toml に保存しておく方法もあります。キーを他人に共有しないでください。",
        placeholder="sk-...",
    ).strip()
    st.caption("ChatGPTの月額プランとは別に、APIの利用料金が発生します。")
    st.divider()
    model_label = st.selectbox("画像モデル", list(MODEL_OPTIONS.keys()), index=0)
    model = MODEL_OPTIONS[model_label]
    size_label = st.selectbox("画像サイズ", list(SIZE_OPTIONS.keys()), index=1)
    size = SIZE_OPTIONS[size_label]
    quality_label = st.selectbox("画質", list(QUALITY_OPTIONS.keys()), index=2)
    quality = QUALITY_OPTIONS[quality_label]
    image_count = st.selectbox("生成枚数（1回あたり）", [1, 2, 3, 4], index=0)
    st.caption("枚数や画質を上げると、料金・待ち時間が増えることがあります。")

st.info("ヒント：構図を守りたい編集では、変えたい箇所と『それ以外は変更しない』条件を具体的に書くと意図が伝わりやすくなります。")

tab_generate, tab_edit = st.tabs(["✏️ 新規生成", "🖼️ 画像を編集"])

with tab_generate:
    st.subheader("テキストから画像を生成")
    prompt = st.text_area(
        "画像の指示文",
        height=160,
        placeholder="例：広告の絵コンテ用のフォトリアルな縦長画像。夕方のスーパーの売り場で、商品を手に取る人物。カメラ位置、照明、構図を具体的に記述する。",
        key="generate_prompt",
    )
    generate_clicked = st.button("画像を生成", type="primary", key="generate_button")

    if generate_clicked:
        if not api_key:
            st.error("APIキーを入力してください。")
        elif not prompt.strip():
            st.error("画像の指示文を入力してください。")
        else:
            try:
                client = OpenAI(api_key=api_key)
                with st.spinner("画像を生成しています。画質や内容によっては少し時間がかかります…"):
                    result = client.images.generate(
                        model=model,
                        prompt=prompt.strip(),
                        size=size,
                        quality=quality,
                        output_format="png",
                        n=image_count,
                    )
                    saved_count = add_results(
                        result, prompt.strip(), model, size, quality, "新規生成"
                    )
                if saved_count:
                    st.success(f"{saved_count}枚の画像を生成しました。下の『生成履歴』から確認・保存できます。")
                else:
                    st.warning("APIから画像データが返りませんでした。設定を変えて再度お試しください。")
            except AuthenticationError:
                st.error("APIキーが認証されませんでした。APIキーが正しいか確認してください。")
            except RateLimitError as exc:
                st.error("APIの利用上限・レート制限に達した可能性があります。APIの請求設定と利用上限を確認してください。")
                st.caption(str(exc))
            except BadRequestError as exc:
                st.error("リクエストが受け付けられませんでした。指示文・画像サイズ・画質などを確認してください。")
                st.caption(str(exc))
            except APIConnectionError:
                st.error("OpenAI APIに接続できませんでした。ネット接続を確認して再度お試しください。")
            except APIStatusError as exc:
                st.error(f"OpenAI API側でエラーが発生しました（HTTP {exc.status_code}）。")
                st.caption(str(exc))
            except APIError as exc:
                st.error("OpenAI APIの処理中にエラーが発生しました。")
                st.caption(str(exc))
            except Exception as exc:
                st.error("予期しないエラーが発生しました。")
                st.caption(f"{type(exc).__name__}: {exc}")

with tab_edit:
    st.subheader("参考画像を指示文に沿って編集")
    st.write("手元の画像をアップロードするか、このアプリで直前までに生成した画像を編集元に選んでください。複数画像を参考資料として指定することもできます。")
    uploaded_images = st.file_uploader(
        "編集元・参考画像（PNG / JPG / WEBP、1ファイル50MB未満）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="edit_uploads",
    )

    history = st.session_state.history
    history_labels = {
        f"{item['created']}｜{item['mode']}｜{item['prompt'][:42].replace(chr(10), ' ')}": item["id"]
        for item in history
    }
    selected_history_labels = st.multiselect(
        "このアプリで生成した画像から編集元・参考画像を追加（任意）",
        options=list(history_labels.keys()),
        help="直前の生成結果を再アップロードせずに編集に使えます。",
        key="edit_history_selection",
    )
    edit_prompt = st.text_area(
        "変更したい内容・維持したい条件",
        height=160,
        placeholder="例：キャラクターの質感だけを2枚目の参考画像に合わせる。構図、顔、ポーズ、背景、商品パッケージの形や位置は一切変更しない。",
        key="edit_prompt",
    )
    edit_clicked = st.button("画像を編集", type="primary", key="edit_button")

    if edit_clicked:
        if not api_key:
            st.error("APIキーを入力してください。")
        elif not edit_prompt.strip():
            st.error("編集内容を入力してください。")
        else:
            api_images = []
            upload_error = None
            for uploaded in uploaded_images or []:
                data = uploaded.getvalue()
                if len(data) >= 50 * 1024 * 1024:
                    upload_error = f"{uploaded.name} は50MB未満のファイルにしてください。"
                    break
                api_images.append(make_api_image(uploaded.name, data, uploaded.type or "image/png"))

            selected_history_ids = {history_labels[label] for label in selected_history_labels}
            for item in history:
                if item["id"] in selected_history_ids:
                    api_images.append(
                        make_api_image(f"history_{item['id']}.png", item["image_bytes"], "image/png")
                    )

            if upload_error:
                st.error(upload_error)
            elif not api_images:
                st.error("編集元・参考画像を少なくとも1枚アップロードするか、生成履歴から選んでください。")
            elif len(api_images) > 16:
                st.error("画像はアップロード分と履歴分を合わせて最大16枚までです。")
            elif not all(len(x[1].getvalue()) < 50 * 1024 * 1024 for x in api_images):
                st.error("参考画像はすべて1ファイル50MB未満にしてください。")
            else:
                try:
                    client = OpenAI(api_key=api_key)
                    with st.spinner("画像を編集しています。参考画像が多い場合は時間がかかることがあります…"):
                        result = client.images.edit(
                            model=model,
                            image=api_images,
                            prompt=edit_prompt.strip(),
                            size=size,
                            quality=quality,
                            output_format="png",
                            n=image_count,
                        )
                        saved_count = add_results(
                            result, edit_prompt.strip(), model, size, quality, "画像編集"
                        )
                    if saved_count:
                        st.success(f"{saved_count}枚の編集画像を作成しました。下の『生成履歴』から確認・保存できます。")
                    else:
                        st.warning("APIから画像データが返りませんでした。設定を変えて再度お試しください。")
                except AuthenticationError:
                    st.error("APIキーが認証されませんでした。APIキーが正しいか確認してください。")
                except RateLimitError as exc:
                    st.error("APIの利用上限・レート制限に達した可能性があります。APIの請求設定と利用上限を確認してください。")
                    st.caption(str(exc))
                except BadRequestError as exc:
                    st.error("リクエストが受け付けられませんでした。画像形式・画像サイズ・指示文などを確認してください。")
                    st.caption(str(exc))
                except APIConnectionError:
                    st.error("OpenAI APIに接続できませんでした。ネット接続を確認して再度お試しください。")
                except APIStatusError as exc:
                    st.error(f"OpenAI API側でエラーが発生しました（HTTP {exc.status_code}）。")
                    st.caption(str(exc))
                except APIError as exc:
                    st.error("OpenAI APIの処理中にエラーが発生しました。")
                    st.caption(str(exc))
                except Exception as exc:
                    st.error("予期しないエラーが発生しました。")
                    st.caption(f"{type(exc).__name__}: {exc}")

st.divider()
st.subheader("生成履歴")
st.caption("履歴はこのブラウザセッションのメモリ上に保持されます。ページやアプリを閉じると消える場合があるため、必要な画像はダウンロードしてください。")

history = st.session_state.history
if history:
    zip_bytes = make_zip(history)
    col_download, col_clear = st.columns([1, 1])
    with col_download:
        st.download_button(
            "履歴の画像をまとめてZIP保存",
            data=zip_bytes,
            file_name="generated_images.zip",
            mime="application/zip",
            key="download_all_zip",
        )
    with col_clear:
        if st.button("履歴をクリア", key="clear_history"):
            st.session_state.history = []
            st.rerun()

    for item in history:
        with st.container(border=True):
            left, right = st.columns([1, 1.2], gap="large")
            with left:
                st.image(item["image_bytes"], use_container_width=True)
                st.download_button(
                    "この画像をPNG保存",
                    data=item["image_bytes"],
                    file_name=f"image_{item['id']}.png",
                    mime="image/png",
                    key=f"download_{item['id']}",
                )
            with right:
                st.markdown(f"**{item['created']}｜{item['mode']}**")
                st.write(item["prompt"])
                st.caption(
                    f"モデル：{item['model']}　|　サイズ：{item['size']}　|　画質：{item['quality']}"
                )
else:
    st.write("まだ画像はありません。上のタブから生成または編集してください。")

st.divider()
st.caption("画像生成にはOpenAI APIの利用料金がかかります。APIの利用料金・組織認証・レート制限はOpenAIの管理画面と公式ドキュメントで確認してください。")
