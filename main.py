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

# 設定 Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 從 GitHub Secrets 環境變數安全地讀取帳號密碼
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# 建立共用的 requests session
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

ASSETS = {
    "🇺🇸 美股大盤 (標普500)": "^GSPC",
    "🇺🇸 科技大盤 (Nasdaq)": "^IXIC",
    "💡 半導體 (費半)": "^SOX",
    "🇹🇼 台股大盤": "^TWII",
    "🚀 台積電": "2330.TW",
    "🇯🇵 日本股市 (日經)": "^N225",
    "🇨🇳 中國股市 (上證)": "000001.SS",
    "🟡 貴金屬 (黃金)": "GC=F",
    "⚪ 貴金屬 (白銀)": "SI=F",
    "📈 美債 ETF (20年+)": "TLT",
    "🪙 比特幣 (BTC)": "BTC-USD",
    "💵 美元指數": "DX-Y.NYB"
}

def get_macro_data():
    logging.info("正在擷取總經數據 (Macro Data)...")
    macro_info = []
    
    try:
        nfci = web.DataReader('NFCI', 'fred')
        nfci_val = round(nfci.iloc[-1][0], 2)
        status = "🔴 資金偏緊縮 (壓力大)" if nfci_val > 0 else "🟢 資金流動性健康 (寬鬆)"
        macro_info.append(f"- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: {nfci_val} ({status})")
    except Exception as e:
        macro_info.append("- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: 擷取失敗")

    try:
        t10y2y = web.DataReader('T10Y2Y', 'fred')
        spread = round(t10y2y.iloc[-1][0], 2)
        status = "⚠️ <b>殖利率倒掛中 (衰退警訊)</b>" if spread < 0 else "✅ 正常斜率 (低衰退疑慮)"
        macro_info.append(f"- 📉 <b>美債 10Y-2Y 利差</b>: {spread}% [{status}]")
    except Exception as e:
        macro_info.append("- 📉 <b>美債 10Y-2Y 利差</b>: 擷取失敗")

    try:
        sahm = web.DataReader('SAHMREALTIME', 'fred')
        sahm_val = round(sahm.iloc[-1][0], 2)
        status = "⚠️ <b>觸發衰退警戒 (失業率飆升)</b>" if sahm_val >= 0.5 else "✅ 就業市場尚穩"
        macro_info.append(f"- 👥 <b>薩姆規則衰退指標</b>: {sahm_val} [{status}]")
    except Exception as e:
        macro_info.append("- 👥 <b>薩姆規則衰退指標</b>: 擷取失敗")

    try:
        spy = yf.Ticker("SPY")
        pe = spy.info.get("trailingPE", 25) 
        dgs10_data = web.DataReader('DGS10', 'fred')
        dgs10 = dgs10_data.dropna().iloc[-1][0]
        
        erp = round((1 / pe) * 100 - dgs10, 2)
        status = "🔴 股市無超額報酬 (風險過高/估值貴)" if erp < 0 else ("✅ 股市風險溢酬佳" if erp >= 2 else "⚪ 估值偏高區間")
        macro_info.append(f"- ⚖️ <b>標普股票風險溢酬 (ERP)</b>: {erp}% [{status}]")
    except Exception as e:
        macro_info.append(f"- ⚖️ <b>標普股票風險溢酬 (ERP)</b>: 擷取失敗")

    return macro_info

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
            
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            hist['STD20'] = hist['Close'].rolling(window=20).std()
            hist['RSI'] = calculate_rsi(hist)

            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            ma20 = hist['MA20'].iloc[-1]
            ma200 = hist['MA200'].iloc[-1]
            std20 = hist['STD20'].iloc[-1]
            rsi = hist['RSI'].iloc[-1]
            pct_change = ((current_price - prev_price) / prev_price) * 100
            
            if ticker == "GC=F":
                gold_price = current_price
            elif ticker == "SI=F":
                silver_price = current_price

            trend = "🟢 多頭 (站上MA20)" if current_price > ma20 else "🔴 空頭 (跌破MA20)"
            
            special_signal = ""
            if ticker == "^GSPC":
                sp500_rsi = rsi
                if not pd.isna(ma200):
                    if current_price > (ma200 * 1.04):
                        special_signal = "\n   🔔 <b>[長線多訊]</b> 突破200MA之上 4% 🚀"
                    elif current_price < (ma200 * 0.97):
                        special_signal = "\n   ⚠️ <b>[長線空訊]</b> 跌破200MA之下 3% 📉"

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
                "動能": f"RSI: {rsi:.1f} ({momentum_text}){special_signal}",
                "型態": pattern_txt
            })
        except Exception as e:
            pass
            
    return results, sp500_rsi, gold_price, silver_price

def scrape_put_call_ratio():
    try:
        url = "https://www.alphaquery.com/data/cboe-put-call-ratio"
        r = session.get(url, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        val_div = soup.find("div", class_="indicator-figure-inner")
        if val_div:
            return float(val_div.text.strip())
    except Exception as e:
        pass
    return None

def get_bottom_signals(sp500_rsi):
    logging.info("正在檢查四大抄底訊號...")
    signals = []
    triggered_count = 0
    
    # 全部換成中文避免被 Telegram 當成網頁語法
    try:
        r = session.get('https://production.dataviz.cnn.io/index/fearandgreed/graphdata', timeout=10)
        fgi = round(r.json().get('fear_and_greed', {}).get('score', 50))
        status = "🔴 達標" if fgi < 10 else "⚪ 未達"
        if fgi < 10: triggered_count += 1
        signals.append(f"1. 恐懼貪婪指數: {fgi} / 門檻小於10 [{status}]")
    except Exception:
        signals.append("1. 恐懼貪婪指數: 擷取失敗")
        
    try:
        vix = round(yf.Ticker("^VIX").history(period="1d")['Close'].iloc[-1], 2)
        status = "🔴 達標" if vix > 40 else "⚪ 未達"
        if vix > 40: triggered_count += 1
        signals.append(f"2. VIX 恐慌指數: {vix} / 門檻大於40 [{status}]")
    except Exception:
        signals.append("2. VIX 恐慌指數: 擷取失敗")

    if sp500_rsi is not None:
        rsi_val = round(sp500_rsi, 1)
        status = "🔴 達標" if rsi_val < 30 else "⚪ 未達"
        if rsi_val < 30: triggered_count += 1
        signals.append(f"3. 標普大盤 RSI : {rsi_val} / 門檻小於30 [{status}]")
    else:
        signals.append("3. 標普大盤 RSI : 資料不足")

    pcr = scrape_put_call_ratio()
    if pcr is not None:
        status = "🔴 達標" if pcr > 0.9 else "⚪ 未達"
        if pcr > 0.9: triggered_count += 1
        signals.append(f"4. Put/Call Ratio: {pcr:.2f} / 門檻大於0.9 [{status}]")
    else:
        signals.append("4. Put/Call Ratio: 擷取失敗 (免付費限制)")

    return signals, triggered_count

def format_telegram_message(market_data, macro_data, bottom_signals, trigger_count):
    today = datetime.now().strftime("%Y-%m-%d")
    msg = f"📊 <b>【全球量化經理人】每日總經早報 ({today})</b>\n\n"
    
    msg += "<b>🌍 =【總經宏觀環境】=</b>\n"
    for item in macro_data:
        msg += f"{item}\n"
    msg += "\n"

    msg += "<b>🛡️ =【四大抄底監控】=</b>\n"
    if trigger_count >= 2:
        msg += "🚨🚨 <b>【強烈抄底訊號提醒】</b> 🚨🚨\n"
        msg += f"<i>目前已有 {trigger_count} 項極端指標觸底！請開始關注進場點！</i>\n"
    elif trigger_count == 1:
        msg += "🚨 <b>【抄底訊號發酵中】</b> (1項達標)\n"
    else:
        msg += "<i>目前處於平靜區間，未見極端超賣。</i>\n"
        
    for sig in bottom_signals:
        msg += f"- {sig}\n"
    msg += "\n"

    msg += "<b>🎯 =【全球核心板塊巡禮】=</b>\n"
    for item in market_data:
        msg += f"<b>{item['名稱']}</b> ({item['代碼']})\n"
        msg += f"   ➤ 價格: {item['目前價格']} ({item['漲跌幅']})\n"
        msg += f"   ➤ 趨勢: {item['趨勢']}\n"
        msg += f"   ➤ 動能: {item['動能']}\n"
        msg += f"   ➤ 型態: {item['型態']}\n"
        msg += f"   ---\n"
        
    msg += "<i>💡 提示: 中長線投資首重總經，機器人波段判讀僅為技術面輔助。</i>"
    return msg

def send_telegram_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
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
    except Exception as e:
        print(f"ERROR: {e}")

def main():
    macro_data = get_macro_data()
    market_data, sp500_rsi, gold_price, silver_price = get_market_data()
    
    if gold_price and silver_price:
        try:
            gs_ratio = gold_price / silver_price
            if gs_ratio > 80:
                status = "🔴 白銀極端便宜 (強烈買入區)"
            elif gs_ratio < 50:
                status = "🟢 黃金極端便宜 (強烈買入區)"
            else:
                status = "⚪ 處於歷史合理區間"
            macro_data.append(f"- 🪙 <b>貴金屬 金銀比 (GSR)</b>: {gs_ratio:.2f} [{status}]")
        except:
            pass

    bottom_signals, trigger_count = get_bottom_signals(sp500_rsi)
    msg = format_telegram
