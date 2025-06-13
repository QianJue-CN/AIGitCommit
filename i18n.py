"""国际化支持模块
支持中英文双语切换
"""

from typing import Dict, Any
import json
from pathlib import Path

# 语言包定义
LANGUAGES = {
    "zh": {
        "app_title": "AI Git 提交助手",
        "page_title": "AI Git Commit Helper",
        "settings": "⚙️ 设置",
        "repository_path": "仓库路径",
        "api_key": "API 密钥",
        "api_url": "API URL (可选)",
        "model": "模型",
        "language": "语言",
        "remember_settings": "记住设置",
        "save_prefs": "💾 保存偏好设置",
        "preferences_saved": "偏好设置已保存",
        "repo_not_exist": "提供的仓库路径不存在",
        "generate_commit": "🚀 生成提交信息",
        "commit": "提交",
        "push": "推送",
        "diff_preview": "🔍 工作区差异预览",
        "no_changes": "未检测到更改 - 无需提交。",
        "api_key_required": "未提供API密钥。",
        "diff_size": "差异大小",
        "tokens": "个令牌",
        "summarizing_diff": "正在总结差异...",
        "summarizing_part": "正在总结第",
        "generating_commit": "正在生成提交信息...",
        "commit_message": "📝 提交信息",
        "committing": "正在执行 git commit...",
        "committed": "已提交",
        "pushing": "正在执行 git push...",
        "pushed": "已推送到远程仓库",
        "llm_provider": "LLM 提供商",
        "custom_model": "自定义模型",
        "model_name": "模型名称",
        "base_url": "基础 URL",
        "temperature": "温度",
        "max_tokens": "最大令牌数",
        "system_prompt": "系统提示词",
        "default_system_prompt": "你是一个助手，负责编写简洁的符合约定式提交规范的提交信息。使用现在时祈使语气；如需要，包含简短描述和要点正文。",
        "openai": "OpenAI",
        "claude": "Claude",
        "local": "本地模型",
        "custom": "自定义",
        "english": "English",
        "chinese": "中文",
        "model_config": "模型配置",
        "advanced_settings": "高级设置",
        "refresh_models": "🔄 刷新模型列表",
        "test_connection": "🔗 测试连接",
        "custom_model_input": "自定义模型名称",
        "model_selection_mode": "模型选择方式",
        "select_from_list": "从列表选择",
        "manual_input": "手动输入",
        "loading_models": "正在加载模型列表...",
        "connection_success": "连接成功",
        "connection_failed": "连接失败",
        "no_models_found": "未找到可用模型",
        "enter_model_name": "请输入模型名称"
    },
    "en": {
        "app_title": "AI Git Commit Helper",
        "page_title": "AI Git Commit Helper", 
        "settings": "⚙️ Settings",
        "repository_path": "Repository Path",
        "api_key": "API Key",
        "api_url": "API URL (optional)",
        "model": "Model",
        "language": "Language",
        "remember_settings": "Remember settings",
        "save_prefs": "💾 Save preferences",
        "preferences_saved": "Preferences saved",
        "repo_not_exist": "Provided repository path does not exist",
        "generate_commit": "🚀 Generate Commit Message",
        "commit": "Commit",
        "push": "Push",
        "diff_preview": "🔍 Workspace Diff Preview",
        "no_changes": "No changes detected – nothing to commit.",
        "api_key_required": "API key not provided.",
        "diff_size": "Diff size",
        "tokens": "tokens",
        "summarizing_diff": "Summarising diff …",
        "summarizing_part": "Summarising part",
        "generating_commit": "Generating commit message …",
        "commit_message": "📝 Commit Message",
        "committing": "Running git commit …",
        "committed": "Committed",
        "pushing": "git push …",
        "pushed": "Pushed to remote",
        "llm_provider": "LLM Provider",
        "custom_model": "Custom Model",
        "model_name": "Model Name",
        "base_url": "Base URL",
        "temperature": "Temperature",
        "max_tokens": "Max Tokens",
        "system_prompt": "System Prompt",
        "default_system_prompt": "You are an assistant that writes concise Conventional Commit messages with scope. Use present‑tense imperative; include short description and body with bullet if needed.",
        "openai": "OpenAI",
        "claude": "Claude", 
        "local": "Local Model",
        "custom": "Custom",
        "english": "English",
        "chinese": "中文",
        "model_config": "Model Configuration",
        "advanced_settings": "Advanced Settings",
        "refresh_models": "🔄 Refresh Models",
        "test_connection": "🔗 Test Connection",
        "custom_model_input": "Custom Model Name",
        "model_selection_mode": "Model Selection Mode",
        "select_from_list": "Select from List",
        "manual_input": "Manual Input",
        "loading_models": "Loading models...",
        "connection_success": "Connection successful",
        "connection_failed": "Connection failed",
        "no_models_found": "No models found",
        "enter_model_name": "Please enter model name"
    }
}

class I18n:
    """国际化类"""
    
    def __init__(self, language: str = "zh"):
        self.language = language
        self.translations = LANGUAGES.get(language, LANGUAGES["zh"])
    
    def set_language(self, language: str):
        """设置语言"""
        if language in LANGUAGES:
            self.language = language
            self.translations = LANGUAGES[language]
    
    def t(self, key: str, **kwargs) -> str:
        """获取翻译文本"""
        text = self.translations.get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError):
                return text
        return text
    
    def get_available_languages(self) -> Dict[str, str]:
        """获取可用语言列表"""
        return {
            "zh": self.t("chinese"),
            "en": self.t("english")
        }

# 全局实例
i18n = I18n()
