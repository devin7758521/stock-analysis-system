import os
import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockAnalyzer:
    def __init__(self):
        self.ai_model = os.getenv('AI_MODEL', 'random_forest')
        self.analysis_days = int(os.getenv('ANALYSIS_DAYS', '30'))
        self.gemini_key = os.getenv('GEMINI_API_KEY')
    
    def preprocess_data(self, stock_data):
        try:
            hist = stock_data['history']
            
            hist['MA5'] = hist['Close'].rolling(window=5).mean()
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            hist['Volume_Change'] = hist['Volume'].pct_change()
            hist['Price_Change'] = hist['Close'].pct_change()
            hist['Volatility'] = hist['Price_Change'].rolling(window=10).std() * (252 ** 0.5)
            
            hist['Momentum'] = hist['Close'] - hist['Close'].shift(5)
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = exp1 - exp2
            hist['Signal'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACD_Hist'] = hist['MACD'] - hist['Signal']
            
            hist['BB_Middle'] = hist['Close'].rolling(window=20).mean()
            hist['BB_Upper'] = hist['BB_Middle'] + 2 * hist['Close'].rolling(window=20).std()
            hist['BB_Lower'] = hist['BB_Middle'] - 2 * hist['Close'].rolling(window=20).std()
            
            hist['Williams_R'] = (hist['High'].rolling(window=14).max() - hist['Close']) / (hist['High'].rolling(window=14).max() - hist['Low'].rolling(window=14).min()) * -100
            
            hist['BIAS5'] = (hist['Close'] - hist['MA5']) / hist['MA5'] * 100
            hist['BIAS20'] = (hist['Close'] - hist['MA20']) / hist['MA20'] * 100
            
            hist = hist.dropna()
            
            return hist
        except Exception as e:
            logger.error(f"预处理数据失败: {e}")
            return None
    
    def prepare_features(self, hist):
        try:
            feature_columns = ['Open', 'High', 'Low', 'Volume', 'MA5', 'MA20', 'MA60', 'Volume_Change', 'Volatility', 'Momentum', 'RSI', 'MACD', 'Signal', 'MACD_Hist', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Williams_R', 'BIAS5', 'BIAS20']
            
            hist['Next_Close'] = hist['Close'].shift(-1)
            hist = hist.dropna()
            
            X = hist[feature_columns]
            y = hist['Next_Close']
            
            return X, y
        except Exception as e:
            logger.error(f"准备特征失败: {e}")
            return None, None
    
    def train_model(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if self.ai_model == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.ai_model == 'linear_regression':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif self.ai_model == 'decision_tree':
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=42)
            elif self.ai_model == 'xgboost':
                try:
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=42)
                except ImportError:
                    logger.warning("XGBoost库未安装，使用随机森林模型")
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"模型训练完成，MSE: {mse:.4f}, R2: {r2:.4f}")
            
            return model, X_test, y_test, y_pred
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            return None, None, None, None
    
    def predict_stock(self, model, X):
        try:
            prediction = model.predict(X)
            return prediction
        except Exception as e:
            logger.error(f"预测股票价格失败: {e}")
            return None
    
    def analyze_stock(self, stock_data):
        try:
            hist = self.preprocess_data(stock_data)
            if hist is None:
                return None
            
            X, y = self.prepare_features(hist)
            if X is None or y is None:
                return None
            
            model, X_test, y_test, y_pred = self.train_model(X, y)
            if model is None:
                return None
            
            last_features = X.iloc[-1:]
            next_price_pred = self.predict_stock(model, last_features)
            
            current_price = stock_data['history']['Close'].iloc[-1]
            predicted_return = (next_price_pred[0] - current_price) / current_price * 100
            
            strategy_result = self.apply_strategy(stock_data['history'])
            
            analysis_result = {
                'symbol': stock_data['symbol'],
                'current_price': current_price,
                'predicted_next_price': next_price_pred[0],
                'predicted_return': predicted_return,
                'model_performance': {
                    'mse': mean_squared_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                },
                'strategy_analysis': strategy_result
            }
            
            recommendation, confidence = self.generate_recommendation(predicted_return, strategy_result)
            analysis_result['recommendation'] = recommendation
            analysis_result['confidence'] = confidence
            
            logger.info(f"股票 {stock_data['symbol']} 分析完成，预测收益率: {predicted_return:.2f}%")
            return analysis_result
        except Exception as e:
            logger.error(f"分析股票失败: {e}")
            return None
    
    def apply_strategy(self, hist):
        """应用策略逻辑：价格(3-70) + 周线量能粘合(-3%到+7%) + 5周均量向上 + 站稳25周线"""
        try:
            CFG_VOL_LOW = -0.03
            CFG_VOL_HIGH = 0.07
            
            result = {
                'price_in_range': False,
                'volume_binding': False,
                'volume_up': False,
                'price_support': False,
                'overall': False
            }
            
            current_price = hist['Close'].iloc[-1]
            result['price_in_range'] = 3.0 <= current_price <= 70.0
            
            df_daily = hist.copy()
            df_daily['week'] = df_daily.index.to_period('W')
            weekly_volumes = df_daily.groupby('week')['Volume'].sum()
            
            logger.info(f"使用原始周量数据，共{len(weekly_volumes)}周")
            
            if len(weekly_volumes) >= 61:
                v5 = weekly_volumes.rolling(5).mean()
                v60 = weekly_volumes.rolling(60).mean()
                
                latest_v5 = v5.iloc[-1]
                latest_v60 = v60.iloc[-1]
                
                if not pd.isna(latest_v5) and not pd.isna(latest_v60) and latest_v60 > 0:
                    result['volume_up'] = latest_v5 > v5.iloc[-2] if len(v5) > 1 else False
                    raw_deviation = (latest_v5 - latest_v60) / latest_v60
                    result['volume_binding'] = CFG_VOL_LOW <= raw_deviation <= CFG_VOL_HIGH
            
            if len(hist) >= 125:
                ma125 = hist['Close'].rolling(125).mean().iloc[-1]
                result['price_support'] = current_price > ma125
            
            result['overall'] = result['price_in_range'] and result['volume_binding'] and result['volume_up'] and result['price_support']
            
            return result
        except Exception as e:
            logger.error(f"应用策略失败: {e}")
            return {
                'price_in_range': False,
                'volume_binding': False,
                'volume_up': False,
                'price_support': False,
                'overall': False
            }
    
    def generate_recommendation(self, predicted_return, strategy_result):
        if strategy_result['overall']:
            return '买入', '高'
        
        if predicted_return > 2:
            return '买入', '高'
        elif predicted_return > 0:
            return '持有', '中'
        else:
            return '卖出', '高'
    
    def analyze_all_stocks(self, stocks_data):
        try:
            analysis_results = []
            
            for symbol, data in stocks_data.items():
                result = self.analyze_stock(data)
                if result:
                    analysis_results.append(result)
            
            analysis_results.sort(key=lambda x: x['predicted_return'], reverse=True)
            
            logger.info(f"成功分析 {len(analysis_results)} 个股票")
            return analysis_results
        except Exception as e:
            logger.error(f"分析所有股票失败: {e}")
            return []
    
    def analyze_with_gemini(self, stocks_data):
        if not self.gemini_key:
            logger.warning("未配置 Gemini API 密钥，跳过 Gemini 分析")
            return None
        
        try:
            now = datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M')
            period_tag = "【早盘观察】" if now.hour < 12 else "【尾盘决策】"
            
            stock_list = []
            for symbol, data in stocks_data.items():
                if 'history' in data and not data['history'].empty:
                    current_price = data['history']['Close'].iloc[-1]
                    stock_name = data['info'].get('name', symbol)
                    stock_list.append(f"- {stock_name}({symbol}): ¥{current_price:.2f}")
            
            if not stock_list:
                logger.error("没有可分析的股票数据")
                return None
            
            api_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={self.gemini_key}"
            
            prompt = (
                f"你是具备全球视野的首席投资官。当前北京时间 {now_str} {period_tag}。\n"
                f"以下是当前分析的股票列表及其价格：\n\n"
                + "\n".join(stock_list)
                + "\n\n"
                f"【决策维度】：\n"
                f"1. **综合评级**：基于当前市场环境，对每只股票进行星级评定（5星严格限制在1-2只）。\n"
                f"2. **短期走势**：预测未来一周走势，给出买入/持有/卖出建议。\n"
                f"3. **风险提示**：指出潜在风险（如政策、业绩）。\n\n"
                f"【输出要求】：\n"
                f"   - 🌟🌟... 股票名(代码) + 30字内深度分析（必须结合当前市场环境）。\n"
                f"   - 未获星标的：仅在下方显示\"代码 名称\"。\n"
            )
            
            response = requests.post(api_url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            ai_text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            logger.info("Gemini API 分析完成")
            return ai_text
        except Exception as e:
            logger.error(f"Gemini API 分析失败: {e}")
            return None
    
    def generate_analysis_report(self, analysis_results):
        try:
            report = "# 股票AI分析报告\n\n"
            
            report += "## 摘要\n"
            buy_count = sum(1 for r in analysis_results if r['recommendation'] == '买入')
            hold_count = sum(1 for r in analysis_results if r['recommendation'] == '持有')
            sell_count = sum(1 for r in analysis_results if r['recommendation'] == '卖出')
            
            report += f"- 分析股票数量: {len(analysis_results)}\n"
            report += f"- 买入建议: {buy_count} 个\n"
            report += f"- 持有建议: {hold_count} 个\n"
            report += f"- 卖出建议: {sell_count} 个\n\n"
            
            report += "## 详细分析\n"
            for result in analysis_results:
                report += f"### {result['symbol']}\n"
                report += f"- 当前价格: ¥{result['current_price']:.2f}\n"
                report += f"- 预测价格: ¥{result['predicted_next_price']:.2f}\n"
                report += f"- 预测收益率: {result['predicted_return']:.2f}%\n"
                report += f"- 建议: {result['recommendation']}\n"
                report += f"- 置信度: {result['confidence']}\n"
                report += f"- 模型R2分数: {result['model_performance']['r2']:.4f}\n"
                
                if 'strategy_analysis' in result:
                    strategy = result['strategy_analysis']
                    report += f"- 策略分析: {'命中' if strategy['overall'] else '未命中'}\n"
                    report += f"  - 价格范围: {'符合' if strategy['price_in_range'] else '不符合'}\n"
                    report += f"  - 量能粘合: {'符合' if strategy['volume_binding'] else '不符合'}\n"
                    report += f"  - 量能向上: {'符合' if strategy['volume_up'] else '不符合'}\n"
                    report += f"  - 站稳均线: {'符合' if strategy['price_support'] else '不符合'}\n"
                
                report += "\n"
            
            return report
        except Exception as e:
            logger.error(f"生成分析报告失败: {e}")
            return "生成分析报告失败"

if __name__ == "__main__":
    from src.github.stock_data import StockDataFetcher
    
    fetcher = StockDataFetcher()
    stocks_data = fetcher.fetch_all_stocks_data()
    
    if stocks_data:
        analyzer = StockAnalyzer()
        analysis_results = analyzer.analyze_all_stocks(stocks_data)
        
        if analysis_results:
            report = analyzer.generate_analysis_report(analysis_results)
            print(report)