import os
import baostock as bs
import pandas as pd
import time
import datetime
from dotenv import load_dotenv
import logging
import sys

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self):
        self.stock_symbols = self.filter_stocks()
        self.analysis_days = int(os.getenv('ANALYSIS_DAYS', '800'))
    
    def print_progress(self, current, total, stock_name='', stage=''):
        if total == 0:
            return
        percent = (current / total) * 100
        bar_length = 30
        filled = int(bar_length * current // total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        msg = f"\r[{bar}] {current}/{total} ({percent:.1f}%)"
        if stock_name:
            msg += f" | 当前: {stock_name}"
        if stage:
            msg += f" | 阶段: {stage}"
        print(msg, end='', flush=True)
    
    def filter_stocks(self):
        try:
            print("\n" + "="*60)
            print("📊 第1步: 开始筛选股票")
            print("="*60)
            
            with self.BaostockContext():
                print("🔗 连接 Baostock API...")
                rs = bs.query_stock_basic()
                stock_list = []
                
                while rs.next():
                    stock = rs.get_row_data()
                    stock_type = stock[2]
                    if stock_type == '1':
                        stock_list.append(stock)
                
                total_stocks = len(stock_list)
                print(f"✅ 获取到 {total_stocks} 只 A 股股票")
                print(f"\n开始筛选 (排除科创板/创业板/北交所/ST/上市不足2年/价格不在3-70元)...\n")
                
                filtered_stocks = []
                stats = {'kcb': 0, 'cyb': 0, 'bse': 0, 'st': 0, 'new': 0, 'price': 0}
                
                for i, stock in enumerate(stock_list):
                    self.print_progress(i + 1, total_stocks, stage='筛选中')
                    
                    try:
                        code = stock[0]
                        name = stock[1]
                        
                        if code.startswith('sh.688'):
                            stats['kcb'] += 1
                            continue
                        
                        if code.startswith('sz.300'):
                            stats['cyb'] += 1
                            continue
                        
                        if code.startswith('bj.8'):
                            stats['bse'] += 1
                            continue
                        
                        if 'ST' in name:
                            stats['st'] += 1
                            continue
                        
                        listing_date = stock[6]
                        if listing_date:
                            listing_datetime = datetime.datetime.strptime(listing_date, '%Y-%m-%d')
                            days_since_listing = (datetime.datetime.now() - listing_datetime).days
                            if days_since_listing < 365 * 2:
                                stats['new'] += 1
                                continue
                        
                        rs_price = bs.query_history_k_data_plus(
                            code, "close",
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
                                    filtered_stocks.append((code, name, price))
                                else:
                                    stats['price'] += 1
                    except Exception as e:
                        continue
                
                print(f"\n\n{'='*60}")
                print("📈 筛选统计:")
                print(f"  • 排除科创板: {stats['kcb']} 只")
                print(f"  • 排除创业板: {stats['cyb']} 只")
                print(f"  • 排除北交所: {stats['bse']} 只")
                print(f"  • 排除ST股票: {stats['st']} 只")
                print(f"  • 排除上市不足2年: {stats['new']} 只")
                print(f"  • 价格不符合(3-70元): {stats['price']} 只")
                print(f"{'='*60}")
                print(f"✅ 符合条件: {len(filtered_stocks)} 只股票")
                
                if filtered_stocks:
                    print(f"\n📋 股票列表:")
                    for code, name, price in filtered_stocks[:20]:
                        print(f"   {code} | {name} | ¥{price:.2f}")
                    if len(filtered_stocks) > 20:
                        print(f"   ... 还有 {len(filtered_stocks)-20} 只")
                
                return [s[0] for s in filtered_stocks]
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
                        time.sleep(2)
                    else:
                        raise net_err
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                bs.logout()
            except Exception:
                pass
    
    def fetch_all_stocks_data(self):
        try:
            total = len(self.stock_symbols)
            
            print("\n" + "="*60)
            print(f"📥 第2步: 获取历史K线数据 (共{total}只股票)")
            print("="*60)
            print(f"⏱️  数据范围: 近{self.analysis_days}个交易日")
            print(f"⏳ 预计耗时: ~{total * 0.10 / 60:.1f} 分钟\n")
            
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            stocks_data = {}
            start_time = time.time()
            
            with self.BaostockContext():
                for idx, symbol in enumerate(self.stock_symbols):
                    self.print_progress(idx + 1, total, symbol.replace('sh.', '').replace('sz.', ''), '获取K线')
                    
                    try:
                        start_date = (datetime.datetime.now() - datetime.timedelta(days=self.analysis_days)).strftime('%Y-%m-%d')
                        
                        rs = bs.query_history_k_data_plus(
                            symbol,
                            "date,open,high,low,close,volume,amount",
                            start_date=start_date,
                            end_date=today,
                            frequency="d",
                            adjustflag="3"
                        )
                        
                        if rs.error_code != '0':
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
                            'open': 'Open', 'high': 'High', 'low': 'Low',
                            'close': 'Close', 'volume': 'Volume', 'amount': 'Amount'
                        }, inplace=True)
                        
                        rs_info = bs.query_stock_basic(code=symbol)
                        info = {'code': symbol, 'name': '', 'industry': '', 'market_cap': 0}
                        if rs_info.error_code == '0':
                            info_list = []
                            while rs_info.next():
                                info_list.append(rs_info.get_row_data())
                            if info_list:
                                row = info_list[0]
                                info = {
                                    'code': row[0], 'name': row[1],
                                    'industry': row[3] if len(row) > 3 else '',
                                    'market_cap': float(row[6]) * 10000 if len(row) > 6 and row[6] else 0
                                }
                        
                        stocks_data[symbol] = {'symbol': symbol, 'history': hist, 'info': info}
                        time.sleep(0.10)
                    except Exception as e:
                        continue
            
            total_time = time.time() - start_time
            print(f"\n\n✅ 数据获取完成! 共 {len(stocks_data)} 只股票, 耗时 {total_time:.1f}s")
            
            return stocks_data
        except Exception as e:
            logger.error(f"获取所有股票数据失败: {e}")
            return {}