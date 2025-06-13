#!/usr/bin/env python3
"""
AI Git Commit Helper 启动脚本
Launch script for AI Git Commit Helper
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'streamlit',
        'git',
        'openai',
        'anthropic',
        'tiktoken',
        'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包 / Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n📦 请运行以下命令安装依赖 / Please run the following command to install dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装 / All dependencies are installed")
    return True

def main():
    """主函数"""
    print("🤖 AI Git Commit Helper")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 获取脚本目录
    script_dir = Path(__file__).parent
    main_script = script_dir / "AIGitCommit.py"
    
    if not main_script.exists():
        print(f"❌ 找不到主脚本: {main_script}")
        sys.exit(1)
    
    # 启动Streamlit应用
    print("🚀 启动应用 / Starting application...")
    print("📱 应用将在浏览器中打开 / Application will open in browser")
    print("🛑 按 Ctrl+C 停止应用 / Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(main_script),
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 应用已停止 / Application stopped")
    except Exception as e:
        print(f"❌ 启动失败 / Failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
