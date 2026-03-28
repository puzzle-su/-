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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip() 
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

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
    logging.info("正在擷取總經數據 (Macro Data)...")
    macro_info = []
    
    try:
        nfci = web.DataReader('NFCI', 'fred')
        nfci_val = round(nfci.iloc[-1, 0], 2)
        status = "🔴 資金偏緊縮 (壓力大)" if nfci_val > 0 else "🟢 資金流動性健康 (寬鬆)"
        macro_info.append(f"- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: {nfci_val} ({status})")
    except Exception as e:
        macro_info.append("- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: 擷取失敗")

    try:
        t10y2y = web.DataReader('T10Y2Y', 'fred')
        spread = round(t10y2y.iloc[-1, 0], 2)
        status = "⚠️ <b>殖利率倒掛中 (衰退警訊)</b>" if spread < 0 else "✅ 正常斜率 (低衰退疑慮)"
        macro_info.append(f"- 📉 <b>美債 10Y-2Y 利差</b>: {spread}% [{status}]")
    except Exception as e:
        macro_info.append("- 📉 <b>美債 10Y-2Y 利差</b>: 擷取失敗")

    try:
        sahm = web.DataReader('SAHMREALTIME', 'fred')
        sahm_val = round(sahm.iloc[-1, 0], 2)
        status = "⚠️ <b>觸發衰退警戒 (失業率飆升)</b>" if sahm_val >= 0.5 else "✅ 就業市場尚穩"
        macro_info.append(f"- 👥 <b>薩姆規則衰退指標</b>: {sahm_val} [{status}]")
    except Exception as e:
        macro_info.append("- 👥 <b>薩姆規則衰退指標</b>: 擷取失敗")

    try:
        spy = yf.Ticker("SPY")
        pe = spy.info.get("trailingPE", 25) 
        dgs10_data = web.DataReader('DGS10', 'fred')
        dgs10 = dgs10_data.dropna().iloc[-1, 0]
        
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
            
            # 1. 計算多天期均線 (20, 60, 200)
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            hist['STD20'] = hist['Close'].rolling(window=20).std()
            hist['RSI'] = calculate_rsi(hist)

            # 2. 計算精準 KD 值 (9, 3, 3) 用於偵測交叉
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

            # 3. 三均線多空排列判定
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

            # 4. KD 黃金交叉或死亡交叉雷達
            kd_signal = " (⚪ KD平行無交叉)"
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

# ==========================================
# 4. 抄底訊號模組 (取代原本易斷線的爬蟲)
# ==========================================
def get_bottom_signals(sp500_rsi):
    logging.info("正在檢查四大抄底訊號...")
    signals = []
    triggered_count = 0
    
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
        signals.append(f"3. 標普大盤 RSI: {rsi_val} / 門檻小於30 [{status}]")
    else:
        signals.append("3. 標普大盤 RSI: 資料不足")

    # 全新引入：終極保命符—VVIX 黑天鵝期權指數 (取代 Put/Call Ratio)
    try:
        vvix = round(yf.Ticker("^VVIX").history(period="1d")['Close'].iloc[-1], 2)
        status = "🔴 達標" if vvix > 115 else "⚪ 未達"
        if vvix > 115: triggered_count += 1
        signals.append(f"4. VVIX 黑天鵝指數: {vvix} / 門檻大於115 [{status}]")
    except Exception:
        signals.append("4. VVIX 黑天鵝指數: 擷取失敗")

    return signals, triggered_count

# ==========================================
# 5. Telegram 推播排版模組
# ==========================================
def format_telegram_message(market_data, macro_data, bottom_signals, trigger_count):
    today = datetime.now().strftime("%Y-%m-%d")
    msg = f"📊 <b>【全球量化經理人】每日總經早報 ({today})</b>\n\n"
    
    # 總經板塊
    msg += "<b>🌍 =【總經宏觀環境】=</b>\n"
    for item in macro_data:
        msg += f"{item}\n"
    msg += "\n"

    # 抄底訊號
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
        error_msg = response.text if response else str(e)
        logging.error(f"Telegram 解析失敗，可能是秘鑰或 ID 格式錯誤!\n詳細錯誤: {error_msg}")

# ==========================================
# 主程式進入點
# ==========================================
def main():
    macro_data = get_macro_data()
    market_data, sp500_rsi, gold_price, silver_price = get_market_data()
    
    # 金銀比估測模組
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

    bottom_signals, trigger_count = get_bottom_signals(sp500_rsi)
    msg = format_telegram_message(market_data, macro_data, bottom_signals, trigger_count)
    send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

if __name__ == "__main__":
    main()
