"""LLM提供商支持模块
支持多种大语言模型提供商
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import requests
import json

class LLMProvider(ABC):
    """LLM提供商基类"""

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """聊天完成接口"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """获取默认模型"""
        pass

    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return []

    def test_connection(self) -> tuple[bool, str]:
        """测试连接，返回(是否成功, 错误信息)"""
        try:
            models = self.get_available_models()
            return True, f"连接成功，找到 {len(models)} 个模型"
        except Exception as e:
            return False, str(e)

class OpenAIProvider(LLMProvider):
    """OpenAI提供商"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        try:
            import openai
            self.openai = openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1"
            )
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.get('model', self.get_default_model())
        temperature = kwargs.get('temperature', 0.2)
        max_tokens = kwargs.get('max_tokens', 400)
        
        backoff = 1
        for _ in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "rate_limit" in str(e).lower():
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise RuntimeError(f"OpenAI API error: {str(e)}")
        raise RuntimeError("OpenAI request failed after retries")
    
    def get_default_model(self) -> str:
        return "gpt-4o-mini"

    def get_available_models(self) -> List[str]:
        """获取OpenAI可用模型列表"""
        try:
            models_response = self.client.models.list()
            models = [model.id for model in models_response.data]
            # 过滤出聊天模型
            chat_models = [m for m in models if any(prefix in m for prefix in ['gpt-', 'text-davinci', 'claude'])]
            return sorted(chat_models)
        except Exception as e:
            # 如果API调用失败，返回预定义列表
            return ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]

class ClaudeProvider(LLMProvider):
    """Claude提供商"""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.get('model', self.get_default_model())
        temperature = kwargs.get('temperature', 0.2)
        max_tokens = kwargs.get('max_tokens', 400)
        
        # 转换消息格式
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=user_messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "claude-3-haiku-20240307"

    def get_available_models(self) -> List[str]:
        """获取Claude可用模型列表"""
        # Claude API目前不提供模型列表接口，返回已知模型
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20241022"
        ]

class LocalProvider(LLMProvider):
    """本地模型提供商（支持Ollama等）"""
    
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.base_url = base_url.rstrip('/')
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.get('model', self.get_default_model())
        temperature = kwargs.get('temperature', 0.2)
        
        # Ollama API格式
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Local model API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "llama2"

    def get_available_models(self) -> List[str]:
        """获取本地模型列表"""
        try:
            # 调用Ollama API获取模型列表
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            models = [model["name"] for model in result.get("models", [])]
            return models
        except Exception:
            # 如果API调用失败，返回常见模型列表
            return ["llama2", "llama3", "codellama", "mistral", "qwen"]

class CustomProvider(LLMProvider):
    """自定义提供商"""
    
    def __init__(self, api_key: str, base_url: str, model: str = "custom-model", **kwargs):
        super().__init__(api_key, base_url, **kwargs)
        self.model = model
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        model = kwargs.get('model', self.model)
        temperature = kwargs.get('temperature', 0.2)
        max_tokens = kwargs.get('max_tokens', 400)
        
        # 通用OpenAI兼容格式
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"Custom API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return self.model

    def get_available_models(self) -> List[str]:
        """获取自定义API的模型列表"""
        try:
            # 尝试调用OpenAI兼容的模型列表接口
            url = f"{self.base_url.rstrip('/')}/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            result = response.json()
            models = [model["id"] for model in result.get("data", [])]
            return models
        except Exception:
            # 如果API调用失败，返回配置的模型
            return [self.model] if self.model else []

def create_llm_provider(provider_type: str, api_key: str, base_url: Optional[str] = None, **kwargs) -> LLMProvider:
    """创建LLM提供商实例"""
    providers = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "local": LocalProvider,
        "custom": CustomProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    
    provider_class = providers[provider_type]
    
    if provider_type == "local":
        return provider_class(base_url=base_url or "http://localhost:11434", **kwargs)
    elif provider_type == "custom":
        if not base_url:
            raise ValueError("Custom provider requires base_url")
        return provider_class(api_key, base_url, **kwargs)
    else:
        return provider_class(api_key, base_url, **kwargs)

# 预定义的模型配置
PROVIDER_CONFIGS = {
    "openai": {
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        "default_base_url": "https://api.openai.com/v1",
        "requires_api_key": True
    },
    "claude": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "default_base_url": "https://api.anthropic.com",
        "requires_api_key": True
    },
    "local": {
        "models": ["llama2", "llama3", "codellama", "mistral", "qwen"],
        "default_base_url": "http://localhost:11434",
        "requires_api_key": False
    },
    "custom": {
        "models": [],
        "default_base_url": "",
        "requires_api_key": True
    }
}
