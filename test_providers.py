#!/usr/bin/env python3
"""
测试LLM提供商功能
Test LLM providers functionality
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_providers import create_llm_provider, PROVIDER_CONFIGS

def test_provider_creation():
    """测试提供商创建"""
    print("🧪 测试提供商创建 / Testing provider creation")
    print("=" * 50)
    
    for provider_type in PROVIDER_CONFIGS.keys():
        try:
            print(f"\n📋 测试 {provider_type} 提供商...")
            
            if provider_type == "local":
                provider = create_llm_provider(provider_type, "", "http://localhost:11434")
            elif provider_type == "custom":
                provider = create_llm_provider(provider_type, "test-key", "http://localhost:8080", model="test-model")
            elif provider_type == "gemini":
                provider = create_llm_provider(provider_type, "test-key")
            else:
                provider = create_llm_provider(provider_type, "test-key")
            
            print(f"✅ {provider_type} 提供商创建成功")
            print(f"   默认模型: {provider.get_default_model()}")
            
        except Exception as e:
            print(f"❌ {provider_type} 提供商创建失败: {e}")

def test_model_listing():
    """测试模型列表获取"""
    print("\n\n🔍 测试模型列表获取 / Testing model listing")
    print("=" * 50)
    
    # 测试本地提供商（不需要API密钥）
    try:
        print("\n📋 测试本地模型列表...")
        provider = create_llm_provider("local", "", "http://localhost:11434")
        models = provider.get_available_models()
        print(f"✅ 找到 {len(models)} 个本地模型:")
        for model in models[:5]:  # 只显示前5个
            print(f"   - {model}")
        if len(models) > 5:
            print(f"   ... 还有 {len(models) - 5} 个模型")
    except Exception as e:
        print(f"❌ 本地模型列表获取失败: {e}")
        print("   (这是正常的，如果没有运行Ollama服务)")

def test_connection():
    """测试连接功能"""
    print("\n\n🔗 测试连接功能 / Testing connection")
    print("=" * 50)
    
    # 测试本地连接
    try:
        print("\n📋 测试本地连接...")
        provider = create_llm_provider("local", "", "http://localhost:11434")
        success, message = provider.test_connection()
        if success:
            print(f"✅ 本地连接成功: {message}")
        else:
            print(f"❌ 本地连接失败: {message}")
    except Exception as e:
        print(f"❌ 本地连接测试失败: {e}")
        print("   (这是正常的，如果没有运行Ollama服务)")

def main():
    """主函数"""
    print("🤖 AI Git Commit Helper - LLM提供商测试")
    print("=" * 60)
    
    test_provider_creation()
    test_model_listing()
    test_connection()
    
    print("\n\n✨ 测试完成 / Testing completed")
    print("💡 提示: 要测试实际的API连接，请配置有效的API密钥")
    print("💡 Tip: To test actual API connections, configure valid API keys")

if __name__ == "__main__":
    main()
