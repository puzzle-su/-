import os
import sys
import logging
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import feedparser
from datetime import datetime
from bs4 import BeautifulSoup
import pandas_datareader.data as web
from scipy.signal import argrelextrema

# 設定 Log，方便我們追蹤程式運作狀態
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 從 GitHub Secrets 環境變數安全地讀取帳號密碼
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "") # 測試環境允許為空
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# 建立共用的 requests session 加上 headers 以避免被阻擋
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# ==========================================
# 1. 定義要追蹤的投資標的 (擴充版)
# ==========================================
ASSETS = {
    "🇺🇸 美股大盤 (標普500)": "^GSPC",
    "🇺🇸 科技大盤 (Nasdaq)": "^IXIC",
    "💡 半導體 (費半)": "^SOX",
    "🇹🇼 台股大盤": "^TWII",
    "🇯🇵 日本股市 (日經)": "^N225",
    "🇨🇳 中國股市 (上證)": "000001.SS",
    "🟡 貴金屬 (黃金)": "GC=F",
    "⚪ 貴金屬 (白銀)": "SI=F",
    "📈 美債 ETF (20年+)": "TLT",
    "🪙 比特幣 (BTC)": "BTC-USD",
    "💵 美元指數": "DX-Y.NYB"
}

# ==========================================
# 2. 總經數據擷取 (FRED)
# ==========================================
def get_macro_data():
    """透過 FRED 取得芝加哥聯儲金融狀況指數與公債殖利率倒掛等總經數據"""
    logging.info("正在擷取總經數據 (Macro Data)...")
    macro_info = []
    
    # 1. 芝加哥聯儲國家金融狀況指數 (NFCI)
    try:
        nfci = web.DataReader('NFCI', 'fred')
        nfci_val = round(nfci.iloc[-1].iloc[0], 2)
        status = "🔴 資金偏緊縮 (壓力大)" if nfci_val > 0 else "🟢 資金流動性健康 (寬鬆)"
        macro_info.append(f"- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: {nfci_val} ({status})")
    except Exception as e:
        macro_info.append("- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: 擷取失敗")

    # 2. 10年期-2年期 公債殖利率利差 (T10Y2Y)
    try:
        t10y2y = web.DataReader('T10Y2Y', 'fred')
        spread = round(t10y2y.iloc[-1].iloc[0], 2)
        status = "⚠️ <b>殖利率倒掛中 (衰退警訊)</b>" if spread < 0 else "✅ 正常斜率 (低衰退疑慮)"
        macro_info.append(f"- 📉 <b>美債 10Y-2Y 利差</b>: {spread}% [{status}]")
    except Exception as e:
        macro_info.append("- 📉 <b>美債 10Y-2Y 利差</b>: 擷取失敗")

    # 3. 薩姆規則衰退指標 (Sahm Rule)
    try:
        sahm = web.DataReader('SAHMREALTIME', 'fred')
        sahm_val = round(sahm.iloc[-1].iloc[0], 2)
        status = "⚠️ <b>觸發衰退警戒 (失業率飆升)</b>" if sahm_val >= 0.5 else "✅ 就業市場尚穩"
        macro_info.append(f"- 👥 <b>薩姆規則衰退指標</b>: {sahm_val} [{status}]")
    except Exception as e:
        macro_info.append("- 👥 <b>薩姆規則衰退指標</b>: 擷取失敗")

    # 4. 股票風險溢酬估測 (ERP) 
    try:
        spy = yf.Ticker("SPY")
        pe = spy.info.get("trailingPE", 25) 
        dgs10_data = web.DataReader('DGS10', 'fred')
        dgs10 = dgs10_data.dropna().iloc[-1].iloc[0]
        
        erp = round((1 / pe) * 100 - dgs10, 2)
        status = "🔴 股市無超額報酬 (風險過高/估值貴)" if erp < 0 else ("✅ 股市風險溢酬佳" if erp >= 2 else "⚪ 估值偏高區間")
        macro_info.append(f"- ⚖️ <b>標普股票風險溢酬 (ERP)</b>: {erp}% [{status}]")
    except Exception as e:
        macro_info.append(f"- ⚖️ <b>標普股票風險溢酬 (ERP)</b>: 擷取失敗")

    return macro_info

# ==========================================
# 3. 核心量化與型態學分析 (Pattern & Wave)
# ==========================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_patterns(hist_df):
    if len(hist_df) < 60:
        return "資料不足"
        
    prices = hist_df['Close'].tail(60).values
    local_maxima = argrelextrema(prices, np.greater, order=3)[0]
    local_minima = argrelextrema(prices, np.less, order=3)[0]
    
    patterns = []
    if len(local_maxima) >= 2 and len(local_minima) >= 1:
        last_two_peaks = prices[local_maxima[-2:]]
        if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.02:
            patterns.append("⚠️ M頭雙頂疑慮")
            
        if len(local_maxima) >= 3 and len(local_minima) >= 3:
            p1, p2, p3 = prices[local_maxima[-3:]]
            v1, v2, v3 = prices[local_minima[-3:]]
            if p3 > p2 > p1 and v3 > v2 > v1:
                patterns.append("🌊 多頭黃金推升波 (推升段)")
            elif p3 < p2 < p1 and v3 < v2 < v1:
                patterns.append("📉 ABC空方修正波 (探底中)")

    if len(local_minima) >= 2 and len(local_maxima) >= 1:
        last_two_valleys = prices[local_minima[-2:]]
        if abs(last_two_valleys[0] - last_two_valleys[1]) / last_two_valleys[0] < 0.02:
            if prices[-1] > last_two_valleys[1] * 1.02:
               patterns.append("🛡️ 強勢 W底成型")
            else:
               patterns.append("🛡️ 正在構築 W底打底")
               
    if not patterns:
        return "⚪ 無明顯大型轉折型態"
        
    return " / ".join(patterns)

def get_market_data():
    results = []
    sp500_rsi = None
    gold_price = None
    silver_price = None
    
    logging.info("正在擷取核心板塊報價與計算技術指標...")
    for name, ticker in ASSETS.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if hist.empty:
                continue
            
            # 計算均線
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            hist['STD20'] = hist['Close'].rolling(window=20).std()
            hist['RSI'] = calculate_rsi(hist)

            # 計算 KD 值 (9, 3, 3)
            hist['Low_9'] = hist['Low'].rolling(window=9).min()
            hist['High_9'] = hist['High'].rolling(window=9).max()
            rsv = (hist['Close'] - hist['Low_9']) / (hist['High_9'] - hist['Low_9']) * 100
            hist['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
            hist['D'] = hist['K'].ewm(alpha=1/3, adjust=False).mean()

            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            ma20 = hist['MA20'].iloc[-1]
            ma60 = hist['MA60'].iloc[-1]
            ma200 = hist['MA200'].iloc[-1]
            std20 = hist['STD20'].iloc[-1]
            rsi = hist['RSI'].iloc[-1]
            k_curr, d_curr = hist['K'].iloc[-1], hist['D'].iloc[-1]
            k_prev, d_prev = hist['K'].iloc[-2], hist['D'].iloc[-2]
            pct_change = ((current_price - prev_price) / prev_price) * 100
            
            if ticker == "GC=F":
                gold_price = current_price
            elif ticker == "SI=F":
                silver_price = current_price

            # 多空均線排列判定
            above_mas = []
            below_mas = []
            if current_price > ma20: above_mas.append("20MA")
            else: below_mas.append("20MA")
            if current_price > ma60: above_mas.append("60MA")
            else: below_mas.append("60MA")
            
            if not pd.isna(ma200):
                if current_price > ma200: above_mas.append("200MA")
                else: below_mas.append("200MA")

            if len(above_mas) == 3 or (len(above_mas) == 2 and pd.isna(ma200)):
                trend = "🟢 多頭排列 (站上全均線)"
            elif len(below_mas) == 3 or (len(below_mas) == 2 and pd.isna(ma200)):
                trend = "🔴 空頭排列 (跌破全均線)"
            else:
                am_str = ",".join(above_mas) if above_mas else "無"
                bm_str = ",".join(below_mas) if below_mas else "無"
                trend = f"🟡 震盪區間 (站上 {am_str} | 破 {bm_str})"

            # KD 交叉判定
            kd_signal = ""
            if k_prev < d_prev and k_curr > d_curr:
                if k_curr < 30:
                    kd_signal = " 💥 低檔KD黃金交叉(抄底)"
                else:
                    kd_signal = " 🔔 KD黃金交叉(偏多)"
            elif k_prev > d_prev and k_curr < d_curr:
                if k_curr > 70:
                    kd_signal = " 💀 高檔KD死亡交叉(逃命)"
                else:
                    kd_signal = " 📉 KD死亡交叉(偏空)"
            
            special_signal = kd_signal
            if ticker == "^GSPC":
                sp500_rsi = rsi

            upper_bb = ma20 + (2 * std20)
            lower_bb = ma20 - (2 * std20)
            
            if rsi < 25 and current_price < lower_bb:
                momentum_text = "🆘 大幅超賣 (跌破布林下軌/乖離過大)"
            elif rsi < 30:
                momentum_text = "⭐ 超賣 (關注買點)"
            elif rsi > 75 and current_price > upper_bb:
                momentum_text = "🔥 大幅超買 (突破布林上軌/竭盡風險)"
            elif rsi > 70:
                momentum_text = "🔥 超買 (短線過熱)"
            else:
                momentum_text = "⚪ 中性"
            
            pattern_txt = analyze_patterns(hist)
            
            results.append({
                "名稱": name,
                "代碼": ticker,
                "目前價格": f"{current_price:.2f}",
                "漲跌幅": f"{pct_change:+.2f}%",
                "趨勢": trend,
                "指標": f"RSI: {rsi:.1f}{special_signal}",
                "型態": pattern_txt
            })
        except Exception as e:
            pass
            
    return results, sp500_rsi, gold_price, silver_price

def get_breadth_data():
    logging.info("正在計算 S5FI 市場寬度指標...")
    s5fi_val = None
    
    def get_wiki_tickers(url, col_name):
        try:
            r = session.get(url, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            if not table: return []
            
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            col_idx = -1
            for i, h in enumerate(headers):
                if col_name.lower() in h:
                    col_idx = i
                    break
            if col_idx == -1: return []
            
            tickers = []
            for row in table.find_all('tr')[1:]:
                cols = row.find_all(['td', 'th'])
                if len(cols) > col_idx:
                    tkr = cols[col_idx].text.strip().replace('.', '-')
                    if tkr: tickers.append(tkr)
            return tickers
        except Exception as e:
            logging.error(f"Failed to fetch tickers: {e}")
            return []

    # 計算 S5FI (標普500成分股中，站上50MA的比例)
    sp500_tickers = get_wiki_tickers('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'symbol')
    if sp500_tickers:
        try:
            data = yf.download(sp500_tickers, period='6mo', threads=True, progress=False)
            if 'Close' in data:
                close_px = data['Close']
                ma50 = close_px.rolling(window=50).mean()
                latest_px = close_px.iloc[-1]
                latest_ma50 = ma50.iloc[-1]
                
                above_50 = (latest_px > latest_ma50).sum()
                total_valid = latest_px.notna().sum()
                if total_valid > 0:
                    s5fi_val = round((above_50 / total_valid) * 100, 2)
        except Exception as e:
            logging.error(f"S5FI Error: {e}")

    return s5fi_val

def get_pcr_5ma():
    logging.info("正在計算 CBOE Put/Call Ratio 5日均線...")
    url = "https://ycharts.com/indicators/cboe_equity_put_call_ratio"
    try:
        r = session.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        pcr_values = []
        tables = soup.find_all('table', class_='table')
        for table in tables:
            headers = [th.text.strip().lower() for th in table.find_all('th')]
            if 'date' in headers and 'value' in headers:
                for row in table.find_all('tr')[1:]:
                    cols = row.find_all('td')
                    if len(cols) == 2:
                        val_str = cols[1].text.strip()
                        try:
                            pcr_values.append(float(val_str))
                        except:
                            continue
                if len(pcr_values) >= 5:
                    break
        if len(pcr_values) >= 5:
            five_day_avg = sum(pcr_values[:5]) / 5.0
            return round(five_day_avg, 2)
    except Exception as e:
        logging.error(f"Failed to fetch PCR: {e}")
    return None

# ==========================================
# 4. 極端訊號模組 (Extreme Signals)
# ==========================================
def get_extreme_signals(sp500_rsi, s5fi_val, pcr_5ma):
    logging.info("正在檢查極端情緒與指標訊號...")
    signals = []
    buy_count = 0
    sell_count = 0
    
    try:
        r = session.get('https://production.dataviz.cnn.io/index/fearandgreed/graphdata', timeout=10)
        fgi = round(r.json().get('fear_and_greed', {}).get('score', 50))
        if fgi < 10:
            status = "🔴 超賣 (買入)"
            buy_count += 1
        elif fgi > 75:
            status = "🟢 超買 (賣出)"
            sell_count += 1
        else:
            status = "⚪ 中性"
        signals.append(f"1. 恐懼貪婪指數: {fgi} / 門檻小於10或大於75 [{status}]")
    except Exception:
        signals.append("1. 恐懼貪婪指數: 擷取失敗")
        
    # 2. VIX 恐慌指數 > 40 或 < 15
    try:
        vix = round(yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1], 2)
        if vix > 40:
            status = "🔴 超賣 (買入)"
            buy_count += 1
        elif vix < 15:
            status = "🟢 超買 (賣出)"
            sell_count += 1
        else:
            status = "⚪ 中性"
        signals.append(f"2. VIX 恐慌指數: {vix} / 門檻大於40或小於15 [{status}]")
    except Exception:
        signals.append("2. VIX 恐慌指數: 擷取失敗")

    # 3. S&P 500 RSI (14日) < 30 或 > 70
    if sp500_rsi is not None:
        rsi_val = round(sp500_rsi, 1)
        if rsi_val < 30:
            status = "🔴 超賣 (買入)"
            buy_count += 1
        elif rsi_val > 70:
            status = "🟢 超買 (賣出)"
            sell_count += 1
        else:
            status = "⚪ 中性"
        signals.append(f"3. 標普大盤 RSI : {rsi_val} / 門檻小於30或大於70 [{status}]")
    else:
        signals.append("3. 標普大盤 RSI : 資料不足")

    # 4. S5FI (標普 50MA 上方比例) < 10% 或 > 85%
    if s5fi_val is not None:
        if s5fi_val < 10:
            status = "🔴 超賣 (買入)"
            buy_count += 1
        elif s5fi_val > 85:
            status = "🟢 超買 (賣出)"
            sell_count += 1
        else:
            status = "⚪ 中性"
        signals.append(f"4. 標普 S5FI: {s5fi_val:.1f}% / 門檻小於10%或大於85% [{status}]")
    else:
        signals.append("4. 標普 S5FI: 擷取失敗")

    # 5. CBOE Put/Call Ratio (5MA) > 0.9 或 < 0.7
    if pcr_5ma is not None:
        if pcr_5ma > 0.9:
            status = "🔴 超賣 (買入)"
            buy_count += 1
        elif pcr_5ma < 0.7:
            status = "🟢 超買 (賣出)"
            sell_count += 1
        else:
            status = "⚪ 中性"
        signals.append(f"5. Put/Call Ratio 5MA: {pcr_5ma:.2f} / 門檻大於0.9或小於0.7 [{status}]")
    else:
        signals.append("5. Put/Call Ratio 5MA: 擷取失敗")

    return signals, buy_count, sell_count

# ==========================================
# 5. Telegram 推播排版模組
# ==========================================
def format_telegram_message(market_data, macro_data, extreme_signals, buy_count, sell_count):
    today = datetime.now().strftime("%Y-%m-%d")
    msg = f"📊 <b>【全球量化經理人】每日總經早報 ({today})</b>\n\n"
    
    # 總經板塊
    msg += "<b>🌍 =【總經宏觀環境】=</b>\n"
    for item in macro_data:
        msg += f"{item}\n"
    msg += "\n"

    # 極端指標訊號
    msg += "<b>🛡️ =【極端市場訊號監控】=</b>\n"
    if buy_count >= 4:
        msg += "🔥🔥🔥 <b>【極端超賣！歷史級別抄底機會】</b> 🔥🔥🔥\n"
        msg += f"<i>目前已有 {buy_count} 項極端超賣指標觸底！強烈建議評估進場！</i>\n"
    elif buy_count >= 2:
        msg += "🚨🚨 <b>【強烈抄底訊號提醒】</b> 🚨🚨\n"
        msg += f"<i>目前已有 {buy_count} 項極端超賣指標觸底！請開始關注進場點！</i>\n"
    elif buy_count == 1:
        msg += "🚨 <b>【抄底訊號發酵中】</b> (1項達標)\n"

    if sell_count >= 4:
        msg += "💀💀💀 <b>【極端超買！泡沫崩盤風險大增】</b> 💀💀💀\n"
        msg += f"<i>目前已有 {sell_count} 項極端超買指標達標！強烈建議減碼或避險！</i>\n"
    elif sell_count >= 2:
        msg += "⚠️⚠️ <b>【高檔過熱訊號提醒】</b> ⚠️⚠️\n"
        msg += f"<i>目前已有 {sell_count} 項極端超買指標達標！注意追高風險！</i>\n"
    elif sell_count == 1:
        msg += "⚠️ <b>【過熱訊號發酵中】</b> (1項達標)\n"
        
    if buy_count == 0 and sell_count == 0:
        msg += "<i>目前處於平靜區間，未見極端市場情緒。</i>\n"
        
    for sig in extreme_signals:
        msg += f"- {sig}\n"
    msg += "\n"

    # 全球行情與型態
    msg += "<b>🎯 =【全球核心板塊巡禮】=</b>\n"
    for item in market_data:
        msg += f"<b>{item['名稱']}</b> ({item['代碼']})\n"
        msg += f"   ➤ 價格: {item['目前價格']} ({item['漲跌幅']})\n"
        msg += f"   ➤ 趨勢: {item['趨勢']}\n"
        msg += f"   ➤ 指標: {item['指標']}\n"
        msg += f"   ➤ 型態: {item['型態']}\n"
        msg += f"   ---\n"
        
    msg += "<i>💡 提示: 中長線投資首重總經，機器人波段判讀僅為技術面輔助。</i>"
    return msg

def send_telegram_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        logging.warning("尚未設定 Bot Token 或 Chat ID，跳過發送並將內容印出於下方：\n" + message)
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        logging.info("✅ 成功發射到使用者的手機囉！")
    except Exception as e:
        logging.error(f"發送 Telegram 訊息失敗: {e}")

# ==========================================
# 主程式進入點
# ==========================================
def main():
    # 判斷是否為獨立的盤中緊急監控模式
    if len(sys.argv) > 1 and sys.argv[1] == '--emergency':
        logging.info("執行盤中緊急監控模式...")
        try:
            hist = yf.Ticker("^GSPC").history(period="1mo")
            hist['RSI'] = calculate_rsi(hist)
            sp500_rsi = hist['RSI'].iloc[-1]
        except:
            sp500_rsi = None
            
        s5fi_val = get_breadth_data()
        pcr_5ma = get_pcr_5ma()
        extreme_signals, buy_count, sell_count = get_extreme_signals(sp500_rsi, s5fi_val, pcr_5ma)
        
        if buy_count >= 1:
            urgent_msg = "🚨🚨🚨 <b>【盤中緊急通知：極端超賣訊號觸發】</b> 🚨🚨🚨\n"
            urgent_msg += f"目前極端監控 5 項指標中，已有 <b>{buy_count} 項</b> 超賣達標！\n\n"
            urgent_msg += "✅ <b>【達標指標】</b>\n"
            for sig in extreme_signals:
                if "🔴 超賣" in sig:
                    urgent_msg += f"{sig}\n"
            urgent_msg += "\n➖ <b>【未達標指標】</b>\n"
            for sig in extreme_signals:
                if "🔴 超賣" not in sig:
                    urgent_msg += f"{sig}\n"
            urgent_msg += "\n請立即開啟看盤軟體評估進場！"
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, urgent_msg)
        elif sell_count >= 1:
            urgent_msg = "💀💀💀 <b>【盤中緊急通知：極端超買訊號觸發】</b> 💀💀💀\n"
            urgent_msg += f"目前極端監控 5 項指標中，已有 <b>{sell_count} 項</b> 超買達標！\n\n"
            urgent_msg += "✅ <b>【達標指標】</b>\n"
            for sig in extreme_signals:
                if "🟢 超買" in sig:
                    urgent_msg += f"{sig}\n"
            urgent_msg += "\n➖ <b>【未達標指標】</b>\n"
            for sig in extreme_signals:
                if "🟢 超買" not in sig:
                    urgent_msg += f"{sig}\n"
            urgent_msg += "\n強烈建議留意風險、適度減碼或避險！"
            send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, urgent_msg)
        else:
            logging.info(f"未達緊急標準。買入達標: {buy_count}，賣出達標: {sell_count}")
        return

    # 以下為原本的每日早報發送流程
    macro_data = get_macro_data()
    market_data, sp500_rsi, gold_price, silver_price = get_market_data()
    
    # 新增：金銀比估測模組 (整合進宏觀數據最後一項)
    if gold_price and silver_price:
        try:
            gs_ratio = gold_price / silver_price
            if gs_ratio > 80:
                status = "🔴 白銀極端便宜 (白銀強烈買入區 / 黃金賣出區)"
            elif gs_ratio < 50:
                status = "🟢 黃金極端便宜 (黃金強烈買入區 / 白銀賣出區)"
            else:
                status = "⚪ 處於歷史合理區間"
            macro_data.append(f"- 🪙 <b>貴金屬 金銀比 (GSR)</b>: {gs_ratio:.2f} [{status}]")
        except:
            pass

    s5fi_val = get_breadth_data()
    pcr_5ma = get_pcr_5ma()
    extreme_signals, buy_count, sell_count = get_extreme_signals(sp500_rsi, s5fi_val, pcr_5ma)
    msg = format_telegram_message(market_data, macro_data, extreme_signals, buy_count, sell_count)
    
    # 判斷是否需要發送緊急獨立通知 (日報附帶)
    if buy_count >= 1:
        urgent_msg = "🔥🔥🔥 <b>【緊急通知：極端超賣訊號觸發】</b> 🔥🔥🔥\n"
        urgent_msg += f"目前極端監控 5 項指標中，已有 <b>{buy_count} 項</b> 超賣達標！\n\n"
        urgent_msg += "✅ <b>【達標指標】</b>\n"
        for sig in extreme_signals:
            if "🔴 超賣" in sig:
                urgent_msg += f"{sig}\n"
        urgent_msg += "\n➖ <b>【未達標指標】</b>\n"
        for sig in extreme_signals:
            if "🔴 超賣" not in sig:
                urgent_msg += f"{sig}\n"
        urgent_msg += "\n請立即評估進場機會！"
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, urgent_msg)
    elif sell_count >= 1:
        urgent_msg = "💀💀💀 <b>【緊急通知：極端超買訊號觸發】</b> 💀💀💀\n"
        urgent_msg += f"目前極端監控 5 項指標中，已有 <b>{sell_count} 項</b> 超買達標！\n\n"
        urgent_msg += "✅ <b>【達標指標】</b>\n"
        for sig in extreme_signals:
            if "🟢 超買" in sig:
                urgent_msg += f"{sig}\n"
        urgent_msg += "\n➖ <b>【未達標指標】</b>\n"
        for sig in extreme_signals:
            if "🟢 超買" not in sig:
                urgent_msg += f"{sig}\n"
        urgent_msg += "\n強烈建議留意風險、適度減碼或避險！"
        send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, urgent_msg)

    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

if __name__ == "__main__":
    main()
