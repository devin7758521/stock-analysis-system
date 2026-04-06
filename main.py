import os
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.github.stock_data import StockDataFetcher
from src.ai.stock_analyzer import StockAnalyzer
from src.wechat.wechat_push import WeChatPusher

class StockAnalysisWorkflow:
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.analyzer = StockAnalyzer()
        self.wechat_pusher = WeChatPusher()
    
    def run(self):
        """运行完整的股票分析工作流"""
        try:
            logger.info("开始股票分析工作流")
            
            # 1. 获取股票数据
            logger.info("正在获取股票数据...")
            stocks_data = self.stock_fetcher.fetch_all_stocks_data()
            
            if not stocks_data:
                logger.error("获取股票数据失败，工作流终止")
                return False
            
            # 2. 分析股票数据
            logger.info("正在分析股票数据...")
            analysis_results = self.analyzer.analyze_all_stocks(stocks_data)
            
            if not analysis_results:
                logger.error("分析股票数据失败，工作流终止")
                return False
            
            # 3. 生成分析报告
            logger.info("正在生成分析报告...")
            report = self.analyzer.generate_analysis_report(analysis_results)
            
            if not report:
                logger.error("生成分析报告失败，工作流终止")
                return False
            
            # 4. 使用 Gemini API 进行深度分析（结合新闻推理）
            logger.info("正在使用 Gemini API 进行新闻推理分析...")
            gemini_analysis = self.analyzer.analyze_with_gemini(stocks_data, analysis_results)
            
            if gemini_analysis:
                report += "## Gemini 新闻推理分析\n"
                report += gemini_analysis + "\n"
            else:
                logger.warning("Gemini API 分析失败，跳过深度分析")
            
            # 5. 推送报告到企业微信
            logger.info("正在推送分析报告到企业微信...")
            success = self.wechat_pusher.send_stock_analysis_report(report)
            
            if success:
                logger.info("股票分析工作流执行成功")
                return True
            else:
                logger.error("推送分析报告失败，工作流终止")
                return False
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            return False

if __name__ == "__main__":
    # 运行工作流
    workflow = StockAnalysisWorkflow()
    success = workflow.run()
    
    if success:
        print("股票分析工作流执行成功！")
    else:
        print("股票分析工作流执行失败！")
