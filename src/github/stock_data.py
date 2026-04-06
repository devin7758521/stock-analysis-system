import os
import baostock as bs
import pandas as pd
import time
import datetime
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self):
        self.stock_symbols = self.filter_stocks()
        self.analysis_days = int(os.getenv('ANALYSIS_DAYS', '800'))
    
    def filter_stocks(self):
        try:
            logger.info("开始筛选股票...")
            
            with self.BaostockContext():
                logger.info("获取所有股票列表...")
                rs = bs.query_stock_basic()
                stock_list = []
                
                while rs.next():
                    stock = rs.get_row_data()
                    stock_type = stock[2]
                    if stock_type == '1':
                        stock_list.append(stock)
                
                logger.info(f"获取到 {len(stock_list)} 只 A 股股票")
                filtered_stocks = []
                processed_count = 0
                max_process = 500
                
                for i, stock in enumerate(stock_list):
                    if processed_count >= max_process:
                        logger.info(f"已处理 {max_process} 只股票，停止处理")
                        break
                    
                    try:
                        code = stock[0]
                        
                        if code.startswith('sh.688'):
                            continue
                        
                        if code.startswith('sz.300'):
                            continue
                        
                        if code.startswith('bj.8'):
                            continue
                        
                        name = stock[1]
                        
                        if 'ST' in name:
                            continue
                        
                        listing_date = stock[6]
                        if listing_date:
                            listing_datetime = datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                            days_since_listing = (datetime.datetime.now() - listing_datetime).days
                            if days_since_listing < 365 * 2:
                                continue
                        
                        processed_count += 1
                        
                        if processed_count % 50 == 0:
                            logger.info(f"处理中... {processed_count}/{max_process}")
                        
                        try:
                            rs_price = bs.query_history_k_data_plus(
                                code,
                                "close",
                                start_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                                end_date=datetime.datetime.now().strftime('%Y-%m-%d'),
                                frequency="d"
                            )
                            
                            if rs_price.error_code == '0':
                                price_data = []
                                while rs_price.next():
                                    price_data.append(rs_price.get_row_data())
                                
                                if price_data:
                                    price = float(price_data[0][0])
                                    if 3.0 <= price <= 70.0:
                                        filtered_stocks.append(code)
                                        logger.info(f"符合条件的股票: {code} - {name} - ¥{price}")
                        except Exception as e:
                            logger.warning(f"获取股票 {code} 价格失败: {e}")
                            continue
                    except Exception as e:
                        logger.error(f"处理股票失败: {e}")
                        continue
                
                logger.info(f"筛选完成，共找到 {len(filtered_stocks)} 只符合条件的股票")
                return filtered_stocks[:50]
        except Exception as e:
            logger.error(f"筛选股票失败: {e}")
            return []
    
    class BaostockContext:
        def __enter__(self):
            for attempt in range(2):
                try:
                    lg = bs.login()
                    if lg.error_code != '0':
                        raise Exception(f"baostock登录失败: {lg.error_msg}")
                    return self
                except Exception as net_err:
                    if "RemoteDisconnected" in str(net_err) or "Connection aborted" in str(net_err):
                        logger.warning(f"baostock登录失败: 网络被断开，休息2秒后重试...")
                        time.sleep(2)
                    else:
                        raise net_err
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                bs.logout()
            except Exception as e:
                logger.warning(f"baostock登出失败: {e}")
    
    def fetch_all_stocks_data(self):
        try:
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            stocks_data = {}
            
            with self.BaostockContext():
                for symbol in self.stock_symbols:
                    try:
                        end_date = today
                        start_date = (datetime.datetime.now() - datetime.timedelta(days=self.analysis_days)).strftime('%Y-%m-%d')
                        
                        rs = bs.query_history_k_data_plus(
                            symbol,
                            "date,open,high,low,close,volume,amount",
                            start_date=start_date,
                            end_date=end_date,
                            frequency="d",
                            adjustflag="3"
                        )
                        
                        if rs.error_code != '0':
                            logger.error(f"获取股票 {symbol} 历史数据失败: {rs.error_msg}")
                            continue
                        
                        data_list = []
                        while rs.next():
                            data_list.append(rs.get_row_data())
                        
                        if not data_list:
                            continue
                        
                        hist = pd.DataFrame(data_list, columns=rs.fields)
                        
                        hist['date'] = pd.to_datetime(hist['date'])
                        hist['open'] = pd.to_numeric(hist['open'])
                        hist['high'] = pd.to_numeric(hist['high'])
                        hist['low'] = pd.to_numeric(hist['low'])
                        hist['close'] = pd.to_numeric(hist['close'])
                        hist['volume'] = pd.to_numeric(hist['volume'])
                        hist['amount'] = pd.to_numeric(hist['amount'])
                        
                        hist.set_index('date', inplace=True)
                        
                        hist.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume',
                            'amount': 'Amount'
                        }, inplace=True)
                        
                        rs_info = bs.query_stock_basic(code=symbol)
                        info = {}
                        if rs_info.error_code == '0':
                            info_list = []
                            while rs_info.next():
                                info_list.append(rs_info.get_row_data())
                            
                            if info_list:
                                info_row = info_list[0]
                                info = {
                                    'code': info_row[0],
                                    'name': info_row[1],
                                    'industry': info_row[3] if len(info_row) > 3 else '',
                                    'market_cap': float(info_row[6]) * 10000 if len(info_row) > 6 and info_row[6] else 0
                                }
                        
                        stocks_data[symbol] = {
                            'symbol': symbol,
                            'history': hist,
                            'info': info
                        }
                        
                        time.sleep(0.15)
                    except Exception as e:
                        logger.error(f"获取股票 {symbol} 数据失败: {e}")
                        continue
            
            logger.info(f"成功获取 {len(stocks_data)} 个股票的数据")
            return stocks_data
        except Exception as e:
            logger.error(f"获取所有股票数据失败: {e}")
            return {}