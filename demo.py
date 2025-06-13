#!/usr/bin/env python3
"""
AI Git Commit Helper 功能演示
Demonstration of AI Git Commit Helper features
"""

import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from llm_providers import create_llm_provider, PROVIDER_CONFIGS
from i18n import i18n

def demo_i18n():
    """演示国际化功能"""
    print("🌐 国际化功能演示 / Internationalization Demo")
    print("=" * 50)
    
    # 中文
    i18n.set_language("zh")
    print(f"\n📋 {i18n.t('language')}: {i18n.t('chinese')}")
    print(f"   {i18n.t('app_title')}")
    print(f"   {i18n.t('generate_commit')}")
    print(f"   {i18n.t('settings')}")
    
    # 英文
    i18n.set_language("en")
    print(f"\n📋 {i18n.t('language')}: {i18n.t('english')}")
    print(f"   {i18n.t('app_title')}")
    print(f"   {i18n.t('generate_commit')}")
    print(f"   {i18n.t('settings')}")

def demo_providers():
    """演示LLM提供商功能"""
    print("\n\n🤖 LLM提供商功能演示 / LLM Provider Demo")
    print("=" * 50)
    
    for provider_type, config in PROVIDER_CONFIGS.items():
        print(f"\n📋 {provider_type.upper()} 提供商:")
        print(f"   需要API密钥: {config['requires_api_key']}")
        print(f"   默认URL: {config['default_base_url']}")
        print(f"   预定义模型: {config['models'][:3]}...")  # 只显示前3个
        
        try:
            if provider_type == "local":
                provider = create_llm_provider(provider_type, "", "http://localhost:11434")
            elif provider_type == "custom":
                provider = create_llm_provider(provider_type, "test-key", "http://localhost:8080", model="test-model")
            else:
                provider = create_llm_provider(provider_type, "test-key")
            
            print(f"   默认模型: {provider.get_default_model()}")
            print(f"   ✅ 提供商创建成功")
            
        except Exception as e:
            print(f"   ❌ 提供商创建失败: {e}")

def demo_model_selection():
    """演示模型选择功能"""
    print("\n\n🔧 模型选择功能演示 / Model Selection Demo")
    print("=" * 50)
    
    # 设置为中文
    i18n.set_language("zh")
    
    print(f"\n📋 {i18n.t('model_selection_mode')}:")
    print(f"   1. {i18n.t('select_from_list')}")
    print(f"   2. {i18n.t('manual_input')}")
    
    print(f"\n🔄 {i18n.t('refresh_models')}")
    print(f"🔗 {i18n.t('test_connection')}")
    
    # 演示本地模型获取
    print(f"\n📋 演示本地模型列表获取:")
    try:
        provider = create_llm_provider("local", "", "http://localhost:11434")
        models = provider.get_available_models()
        if models:
            print(f"   ✅ 找到 {len(models)} 个模型:")
            for model in models[:5]:
                print(f"      - {model}")
        else:
            print(f"   ⚠️ 未找到模型（可能Ollama未运行）")
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        print(f"   💡 提示: 请确保Ollama服务正在运行")

def demo_features():
    """演示主要功能"""
    print("\n\n✨ 主要功能特性 / Key Features")
    print("=" * 50)
    
    features = [
        "🌐 中英文双语界面",
        "🤖 多LLM提供商支持 (OpenAI, Claude, 本地模型, 自定义)",
        "🔧 动态模型列表获取",
        "✏️ 手动模型名称输入",
        "🔗 API连接测试",
        "📝 智能提交信息生成",
        "⚙️ 灵活的配置选项",
        "💾 配置持久化存储"
    ]
    
    for feature in features:
        print(f"   {feature}")

def main():
    """主函数"""
    print("🤖 AI Git Commit Helper - 功能演示")
    print("AI Git Commit Helper - Feature Demonstration")
    print("=" * 60)
    
    demo_i18n()
    demo_providers()
    demo_model_selection()
    demo_features()
    
    print("\n\n🚀 启动应用 / Start Application")
    print("=" * 50)
    print("运行以下命令启动应用:")
    print("Run the following command to start the application:")
    print()
    print("   python run.py")
    print("   或者 / or")
    print("   streamlit run AIGitCommit.py")
    
    print("\n💡 提示 / Tips:")
    print("- 配置有效的API密钥以测试实际功能")
    print("- Configure valid API keys to test actual functionality")
    print("- 确保Git仓库路径正确")
    print("- Ensure Git repository path is correct")

if __name__ == "__main__":
    main()
