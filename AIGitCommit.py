"""AIGitCommit – Streamlit Git helper with Multi-LLM commit‑message generation
=======================================================================
Features implemented
--------------------
* Multi-LLM provider support (OpenAI, Claude, Local models, Custom APIs).
* Chinese/English bilingual interface with language switching.
* Custom `api_base` (apiURL) support.
* Accurate token pre‑estimation; chunks and summarises long diffs automatically.
* Detects whether the repo has remotes and greys‑out **Push** option if none.
* All subprocess calls use UTF‑8 decoding with error replacement.
* Multi‑stage progress bar and diff preview in expandable panel.
* Diff collection covers **added / modified / deleted** (tracked + untracked).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from base64 import b64decode, b64encode
from pathlib import Path
from typing import Iterable, List, Optional

import streamlit as st

try:
    import git  # GitPython
except ImportError:
    st.error("GitPython is required: pip install GitPython")
    sys.exit(1)

try:
    import tiktoken  # optional but preferred
except ImportError:  # graceful degradation
    tiktoken = None  # type: ignore

# 导入新模块
from i18n import i18n
from llm_providers import create_llm_provider, PROVIDER_CONFIGS

CONFIG_PATH = Path.home() / ".ai_commit_helper_config.json"
DEFAULT_MODEL = "gpt-4o-mini"
MODEL_MAX_TOKENS = 32000  # default max context
CHUNK_TOKEN_TARGET = 2000  # ~2k token per chunk when summarising


# ---------------------------------------------------------------------------
# 🔧 Utility helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.is_file():
        data = json.loads(CONFIG_PATH.read_text())
        if "api_key" in data:
            try:
                data["api_key"] = b64decode(data["api_key"]).decode()
            except Exception:
                data.pop("api_key")
        return data
    return {
        "language": "zh",  # 默认中文
        "llm_provider": "openai",
        "model": DEFAULT_MODEL,
        "temperature": 0.2,
        "max_tokens": 400
    }


def save_config(cfg: dict) -> None:
    cp = cfg.copy()
    if "api_key" in cp:
        cp["api_key"] = b64encode(cp["api_key"].encode()).decode()
    CONFIG_PATH.write_text(json.dumps(cp, indent=2))


# ---------- Git helpers -----------------------------------------------------


def run_git(repo_dir: Path, args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        "git",
        "-C",
        str(repo_dir),
        *args,
    ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def repo_has_remote(repo: git.Repo) -> bool:
    return bool(repo.remotes)


def get_status_porcelain(repo_dir: Path) -> List[str]:
    p = run_git(repo_dir, ["status", "--porcelain"])
    return p.stdout.splitlines()


def collect_workspace_diff(repo_dir: Path) -> str:
    """Return unified diff for added/modified/deleted (tracked + untracked)."""
    diff_parts: List[str] = []

    # 1. staged & unstaged tracked changes (add / mod / del)
    diff_parts.append(run_git(repo_dir, ["diff"]).stdout)

    # 2. untracked files – diff against /dev/null
    status = get_status_porcelain(repo_dir)
    for line in status:
        if line and line[0] == "?":
            filepath = line[3:]
            diff_untracked = run_git(repo_dir, [
                "diff",
                "--no-index",
                "--",
                os.devnull,
                filepath,
            ]).stdout
            diff_parts.append(diff_untracked)

    return "\n".join(part for part in diff_parts if part.strip())


# ---------- Token helpers ---------------------------------------------------


def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    if tiktoken is None:
        return len(text) // 4  # crude fallback
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except KeyError:
        # 如果模型不支持，使用默认编码
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int) -> List[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # naive line‑based chunking to stay below limit
    lines = text.splitlines(keepends=True)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tok = 0
    for ln in lines:
        t = estimate_tokens(ln)
        if cur_tok + t > max_tokens and cur:
            chunks.append("".join(cur))
            cur, cur_tok = [], 0
        cur.append(ln)
        cur_tok += t
    if cur:
        chunks.append("".join(cur))
    return chunks


# ---------- LLM interaction ---------------------------------------------

def create_llm_chat_function(provider_type: str, api_key: str, base_url: Optional[str] = None, **config):
    """创建LLM聊天函数"""
    provider = create_llm_provider(provider_type, api_key, base_url, **config)

    def llm_chat(messages: List[dict], **kwargs) -> str:
        return provider.chat_completion(messages, **kwargs)

    return llm_chat


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()

    # 设置语言
    language = cfg.get("language", "zh")
    i18n.set_language(language)

    st.set_page_config(
        page_title=i18n.t("page_title"),
        page_icon="🤖",
        layout="wide"
    )

    # ------------- Settings form ------------------------------------------
    with st.sidebar:
        st.header(i18n.t("settings"))

        # 语言选择
        available_languages = i18n.get_available_languages()
        language_options = list(available_languages.keys())
        language_labels = list(available_languages.values())
        current_lang_index = language_options.index(language) if language in language_options else 0

        selected_language = st.selectbox(
            i18n.t("language"),
            options=language_options,
            format_func=lambda x: available_languages[x],
            index=current_lang_index,
            key="language_selector"
        )

        # 如果语言改变，更新i18n并重新运行
        if selected_language != language:
            cfg["language"] = selected_language
            save_config(cfg)
            st.rerun()

        # LLM提供商选择（在表单外）
        provider_options = list(PROVIDER_CONFIGS.keys())
        current_provider = cfg.get("llm_provider", "openai")
        provider_index = provider_options.index(current_provider) if current_provider in provider_options else 0

        llm_provider = st.selectbox(
            i18n.t("llm_provider"),
            options=provider_options,
            format_func=lambda x: i18n.t(x),
            index=provider_index,
            key="provider_selector"
        )

        # 根据选择的提供商显示相应配置
        provider_config = PROVIDER_CONFIGS[llm_provider]

        # API配置（在表单外）
        if provider_config["requires_api_key"]:
            api_key = st.text_input(i18n.t("api_key"), cfg.get("api_key", ""), type="password", key="api_key_input")
        else:
            api_key = ""

        # API URL配置
        default_url = provider_config.get("default_base_url", "")
        api_url = st.text_input(
            i18n.t("base_url"),
            cfg.get("api_url", default_url),
            help=f"默认: {default_url}" if default_url else None,
            key="api_url_input"
        )

        # 模型选择方式
        model_selection_mode = st.radio(
            i18n.t("model_selection_mode"),
            options=["select_from_list", "manual_input"],
            format_func=lambda x: i18n.t(x),
            horizontal=True,
            key="model_selection_mode"
        )

        # 测试连接和刷新模型按钮（在表单外）
        col1, col2 = st.columns(2)
        with col1:
            test_conn_btn = st.button(i18n.t("test_connection"), key="test_connection")
        with col2:
            refresh_models_btn = st.button(i18n.t("refresh_models"), key="refresh_models")

        # 处理测试连接
        if test_conn_btn:
            if provider_config["requires_api_key"] and not api_key:
                st.error(i18n.t("api_key_required"))
            else:
                try:
                    with st.spinner(i18n.t("loading_models")):
                        provider = create_llm_provider(llm_provider, api_key, api_url or None)
                        success, message = provider.test_connection()
                        if success:
                            st.success(f"{i18n.t('connection_success')}: {message}")
                        else:
                            st.error(f"{i18n.t('connection_failed')}: {message}")
                except Exception as e:
                    st.error(f"{i18n.t('connection_failed')}: {str(e)}")

        # 模型选择
        if model_selection_mode == "manual_input":
            model_name = st.text_input(
                i18n.t("custom_model_input"),
                cfg.get("model", ""),
                placeholder=i18n.t("enter_model_name"),
                key="manual_model_input"
            )
        else:
            # 从列表选择
            available_models = provider_config["models"]

            # 如果点击刷新或者模型列表为空，尝试获取实时模型列表
            if refresh_models_btn or not available_models:
                if provider_config["requires_api_key"] and not api_key:
                    st.warning(i18n.t("api_key_required"))
                    available_models = provider_config["models"]
                else:
                    try:
                        with st.spinner(i18n.t("loading_models")):
                            provider = create_llm_provider(llm_provider, api_key, api_url or None)
                            fetched_models = provider.get_available_models()
                            if fetched_models:
                                available_models = fetched_models
                                st.success(f"✅ {i18n.t('connection_success')}: 找到 {len(fetched_models)} 个模型")
                                # 将获取的模型列表存储到session state中
                                st.session_state[f"models_{llm_provider}"] = fetched_models
                            else:
                                st.warning(i18n.t("no_models_found"))
                                available_models = provider_config["models"]
                    except Exception as e:
                        st.error(f"{i18n.t('connection_failed')}: {str(e)}")
                        available_models = provider_config["models"]

            # 尝试从session state获取之前获取的模型列表
            if f"models_{llm_provider}" in st.session_state:
                cached_models = st.session_state[f"models_{llm_provider}"]
                if cached_models:
                    available_models = cached_models

            if available_models:
                current_model = cfg.get("model", available_models[0])
                model_index = 0
                if current_model in available_models:
                    model_index = available_models.index(current_model)
                model_name = st.selectbox(
                    i18n.t("model"),
                    available_models,
                    index=model_index,
                    key="model_selectbox"
                )
            else:
                st.warning(i18n.t("no_models_found"))
                model_name = st.text_input(
                    i18n.t("custom_model_input"),
                    cfg.get("model", ""),
                    placeholder=i18n.t("enter_model_name"),
                    key="fallback_model_input"
                )

        with st.form("settings"):
            repo_dir = st.text_input(i18n.t("repository_path"), cfg.get("repo_dir", str(Path.cwd())))

            # 高级设置
            with st.expander(i18n.t("advanced_settings")):
                temperature = st.slider(i18n.t("temperature"), 0.0, 1.0, cfg.get("temperature", 0.2), 0.1)
                max_tokens = st.number_input(i18n.t("max_tokens"), 100, 4000, cfg.get("max_tokens", 400))
                system_prompt = st.text_area(
                    i18n.t("system_prompt"),
                    cfg.get("system_prompt", i18n.t("default_system_prompt")),
                    height=100
                )

            remember_settings = st.checkbox(i18n.t("remember_settings"), value=cfg.get("remember", True))
            save_pref_btn = st.form_submit_button(i18n.t("save_prefs"))

        if save_pref_btn and remember_settings:
            save_config({
                "repo_dir": repo_dir,
                "api_key": api_key,
                "api_url": api_url,
                "llm_provider": llm_provider,
                "model": model_name,
                "model_selection_mode": model_selection_mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "language": selected_language,
                "remember": remember_settings,
            })
            st.success(i18n.t("preferences_saved"))

    if not Path(repo_dir).is_dir():
        st.error(i18n.t("repo_not_exist"))
        return

    repo = git.Repo(repo_dir)
    has_remote = repo_has_remote(repo)

    st.title(i18n.t("app_title") + " 🤖✍️")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        gen_btn = st.button(i18n.t("generate_commit"))
    with col2:
        commit_chk = st.checkbox(i18n.t("commit"), value=True)
    with col3:
        push_chk = st.checkbox(i18n.t("push"), value=False, disabled=not has_remote)

    # diff collection & preview ------------------------------------------------
    with st.spinner(i18n.t("summarizing_diff")):
        diff_text = collect_workspace_diff(Path(repo_dir))

    with st.expander(i18n.t("diff_preview")):
        st.code(diff_text or "<empty>", language="diff")

    if not gen_btn:
        return

    if not diff_text.strip():
        st.warning(i18n.t("no_changes"))
        return

    # 检查API密钥（某些提供商不需要）
    if PROVIDER_CONFIGS[llm_provider]["requires_api_key"] and not api_key:
        st.error(i18n.t("api_key_required"))
        return

    # 创建LLM聊天函数
    try:
        llm_chat = create_llm_chat_function(
            llm_provider,
            api_key,
            api_url or None,
            model=model_name
        )
    except Exception as e:
        st.error(f"LLM配置错误: {str(e)}")
        return

    # ------------- Token estimation & possible summarisation ---------------
    diff_tokens = estimate_tokens(diff_text, model_name)
    st.info(f"{i18n.t('diff_size')}: ~{diff_tokens} {i18n.t('tokens')}")

    # 构建系统提示词
    sys_prompt = {
        "role": "system",
        "content": system_prompt,
    }

    chunks = chunk_text(diff_text, CHUNK_TOKEN_TARGET)

    summary_sections: List[str] = []
    if len(chunks) > 1:
        pbar = st.progress(0, text=i18n.t("summarizing_diff"))
        for idx, chunk in enumerate(chunks):
            pbar.progress(
                (idx + 1) / len(chunks),
                text=f"{i18n.t('summarizing_part')} {idx + 1}/{len(chunks)}"
            )
            summary = llm_chat([
                sys_prompt,
                {"role": "user", "content": f"Summarise the following git diff:\n\n{chunk}"},
            ], model=model_name, temperature=temperature, max_tokens=max_tokens)
            summary_sections.append(summary)
        pbar.empty()

    final_prompt = (
        "\n\n".join(summary_sections)
        if len(chunks) > 1 else diff_text
    )

    with st.spinner(i18n.t("generating_commit")):
        commit_msg = llm_chat([
            sys_prompt,
            {"role": "user", "content": f"Write a conventional commit message for this summary:\n\n{final_prompt}"},
        ], model=model_name, temperature=temperature, max_tokens=max_tokens)

    st.text_area(i18n.t("commit_message"), commit_msg, height=200)

    # ------------- Commit / push -------------------------------------------
    if commit_chk:
        with st.spinner(i18n.t("committing")):
            run_git(Path(repo_dir), ["add", "--all"])
            cp = run_git(Path(repo_dir), ["commit", "-m", commit_msg])
            if cp.returncode != 0:
                st.error(cp.stderr)
                return
            st.success(i18n.t("committed"))
    if push_chk:
        with st.spinner(i18n.t("pushing")):
            cp = run_git(Path(repo_dir), ["push"])
            if cp.returncode != 0:
                st.error(cp.stderr)
                return
            st.success(i18n.t("pushed"))

    # Save config after successful run
    if remember_settings:
        save_config({
            "repo_dir": repo_dir,
            "api_key": api_key,
            "api_url": api_url,
            "llm_provider": llm_provider,
            "model": model_name,
            "model_selection_mode": model_selection_mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "language": selected_language,
            "remember": remember_settings,
        })


if __name__ == "__main__":
    main()
