# AI Git Commit Helper / AI Git 提交助手

一个支持多种大语言模型的智能Git提交信息生成工具，具有中英文双语界面。

A smart Git commit message generator supporting multiple LLM providers with bilingual Chinese/English interface.

## 功能特性 / Features

### 🌐 多语言支持 / Multi-language Support
- 中英文双语界面切换
- Bilingual Chinese/English interface switching

### 🤖 多LLM提供商支持 / Multi-LLM Provider Support
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo等
- **Claude**: Claude-3 系列模型
- **本地模型**: 支持Ollama等本地部署模型
- **自定义API**: 支持任何OpenAI兼容的API

### 📝 智能提交信息生成 / Smart Commit Message Generation
- 自动分析Git差异
- 符合约定式提交规范
- 支持大型差异的自动分块和总结
- 可自定义系统提示词

### ⚙️ 灵活配置 / Flexible Configuration
- 模型参数调节（温度、最大令牌数等）
- API端点自定义
- 配置持久化存储

### 🔧 增强的模型选择 / Enhanced Model Selection
- **动态模型列表**: 自动从API获取可用模型
- **手动输入**: 支持手动输入自定义模型名称
- **连接测试**: 一键测试API连接状态
- **模型刷新**: 实时更新可用模型列表

## 安装 / Installation

1. 克隆仓库 / Clone repository:
```bash
git clone git@github.com:QianJue-CN/AIGitCommit.git
cd Python/AICommit
```

2. 安装依赖 / Install dependencies:
```bash
pip install -r requirements.txt
```

## 使用方法 / Usage

1. 启动应用 / Start the application:
```bash
streamlit run AIGitCommit.py
```

2. 配置设置 / Configure settings:
   - 选择语言 / Select language
   - 选择LLM提供商 / Choose LLM provider
   - 输入API密钥（如需要）/ Enter API key (if required)
   - 选择模型选择方式（列表选择或手动输入）/ Choose model selection mode
   - 测试连接并刷新模型列表 / Test connection and refresh model list
   - 设置仓库路径 / Set repository path

3. 生成提交信息 / Generate commit message:
   - 点击"生成提交信息"按钮
   - 查看生成的提交信息
   - 选择是否直接提交和推送

## 支持的LLM提供商 / Supported LLM Providers

### OpenAI
- 模型: gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo
- 需要API密钥
- 默认端点: https://api.openai.com/v1

### Claude (Anthropic)
- 模型: claude-3-opus, claude-3-sonnet, claude-3-haiku
- 需要API密钥
- 默认端点: https://api.anthropic.com

### 本地模型 (Ollama)
- 支持任何Ollama部署的模型
- 无需API密钥
- 默认端点: http://localhost:11434

### 自定义API
- 支持任何OpenAI兼容的API
- 需要配置API密钥和端点

## 配置文件 / Configuration

配置文件保存在 `~/.ai_commit_helper_config.json`

Configuration is saved in `~/.ai_commit_helper_config.json`

## 依赖项 / Dependencies

- streamlit: Web界面框架
- GitPython: Git操作
- openai: OpenAI API客户端
- anthropic: Claude API客户端
- tiktoken: 令牌计算
- requests: HTTP请求

## 许可证 / License

MIT License

## 贡献 / Contributing

欢迎提交Issue和Pull Request！

Issues and Pull Requests are welcome!
