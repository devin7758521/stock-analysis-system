# A股股票分析系统

自动筛选 A 股股票 + AI 深度分析 + 企业微信报告推送

## 功能特点

- 📊 **真实数据**: 使用 Baostock API 获取实时股票数据
- 🔍 **智能筛选**: 自动筛选 3-70 元的 A 股，排除 ST/科创板/创业板/北交所
- 🤖 **AI 分析**: 使用 Google Gemini 2.5 Flash 进行深度分析
- 📱 **企业微信推送**: 通过 Webhook 自动推送分析报告
- ⏰ **定时运行**: 通过 GitHub Actions 每天自动执行

## 使用方法

### 本地运行

```bash
pip install -r requirements.txt
python main.py
```

### GitHub 部署

1. Fork 本仓库
2. 在 Settings → Secrets 中配置以下密钥:
   - `GEMINI_API_KEY`: Google Gemini API 密钥
   - `WECHAT_WEBHOOK_URL`: 企业微信 Webhook URL
3. 在 Actions 页面手动触发或等待定时运行

## 免责声明

本系统仅供学习参考，不构成投资建议。股市有风险，投资需谨慎。