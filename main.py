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
        nfci_val = round(nfci.iloc[-1][0], 2)
        status = "🔴 資金偏緊縮 (壓力大)" if nfci_val > 0 else "🟢 資金流動性健康 (寬鬆)"
        macro_info.append(f"- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: {nfci_val} ({status})")
    except Exception as e:
        macro_info.append("- 🏦 <b>聯儲金融狀況指數 (NFCI)</b>: 擷取失敗")

    # 2. 10年期-2年期 公債殖利率利差 (T10Y2Y)
    try:
        t10y2y = web.DataReader('T10Y2Y', 'fred')
        spread = round(t10y2y.iloc[-1][0], 2)
        status = "⚠️ <b>殖利率倒掛中 (衰退警訊)</b>" if spread < 0 else "✅ 正常斜率 (低衰退疑慮)"
        macro_info.append(f"- 📉 <b>美債 10Y-2Y 利差</b>: {spread}% [{status}]")
    except Exception as
