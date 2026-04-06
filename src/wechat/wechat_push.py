import os
import requests
import json
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeChatPusher:
    def __init__(self):
        self.webhook_key = os.getenv('WECHAT_WEBHOOK_KEY')
        self.to_user = os.getenv('WECHAT_TO_USER', '@all')
        self.corp_id = os.getenv('WECHAT_CORP_ID')
        self.app_secret = os.getenv('WECHAT_APP_SECRET')
        self.agent_id = os.getenv('WECHAT_AGENT_ID')
        self.token_url = f'https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid={self.corp_id}&corpsecret={self.app_secret}'
        self.message_url = 'https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token='
    
    def send_wechat_webhook(self, content):
        if not self.webhook_key:
            logger.warning("未配置企业微信 webhook 密钥")
            return False
        
        try:
            url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={self.webhook_key}"
            response = requests.post(url, json={"msgtype": "markdown", "markdown": {"content": content}}, timeout=10)
            result = response.json()
            
            if result['errcode'] == 0:
                logger.info("成功发送企业微信 webhook 消息")
                return True
            else:
                logger.error(f"发送企业微信 webhook 消息失败: {result['errmsg']}")
                return False
        except Exception as e:
            logger.error(f"发送企业微信 webhook 消息失败: {e}")
            return False
    
    def get_access_token(self):
        try:
            response = requests.get(self.token_url)
            result = response.json()
            
            if result['errcode'] == 0:
                logger.info("成功获取企业微信访问令牌")
                return result['access_token']
            else:
                logger.error(f"获取企业微信访问令牌失败: {result['errmsg']}")
                return None
        except Exception as e:
            logger.error(f"获取企业微信访问令牌失败: {e}")
            return None
    
    def send_markdown_message(self, content):
        try:
            if self.webhook_key:
                return self.send_wechat_webhook(content)
            
            access_token = self.get_access_token()
            if not access_token:
                return False
            
            message_data = {
                "touser": self.to_user,
                "agentid": self.agent_id,
                "msgtype": "markdown",
                "markdown": {"content": content}
            }
            
            url = self.message_url + access_token
            response = requests.post(url, data=json.dumps(message_data), headers={'Content-Type': 'application/json'})
            result = response.json()
            
            if result['errcode'] == 0:
                logger.info("成功发送企业微信Markdown消息")
                return True
            else:
                logger.error(f"发送企业微信Markdown消息失败: {result['errmsg']}")
                return False
        except Exception as e:
            logger.error(f"发送企业微信Markdown消息失败: {e}")
            return False
    
    def send_stock_analysis_report(self, report):
        try:
            success = self.send_markdown_message(report)
            
            if success:
                logger.info("成功发送股票分析报告到企业微信")
                return True
            else:
                logger.error("发送股票分析报告失败")
                return False
        except Exception as e:
            logger.error(f"发送股票分析报告失败: {e}")
            return False