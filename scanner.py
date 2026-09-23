import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import requests

warnings.filterwarnings('ignore')

def get_all_assets_tickers():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    nasdaq_tickers = []
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns:
                nasdaq_tickers = table['Ticker'].tolist()
                break
            elif 'Symbol' in table.columns:
                nasdaq_tickers = table['Symbol'].tolist()
                break
    except Exception:
        nasdaq_tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NFLX', 'AMD', 'INTC']
    
    crypto_tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'XAUT-USD']
    return nasdaq_tickers + crypto_tickers

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_timeframe(df, period_name):
    if len(df) < 50:
        return None
    
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_31'] = df['Close'].ewm(span=31, adjust=False).mean()
    df['EMA_150'] = df['Close'].ewm(span=150, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    last_row = df.iloc[-1]
    
    def clean_val(val):
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        return float(val)

    price = clean_val(last_row['Close'])
    ema_21 = clean_val(last_row['EMA_21'])
    ema_31 = clean_val(last_row['EMA_31'])
    ema_150 = clean_val(last_row['EMA_150'] if not pd.isna(last_row['EMA_150']) else price)
    ema_200 = clean_val(last_row['EMA_200'] if not pd.isna(last_row['EMA_200']) else price)
    rsi = clean_val(last_row['RSI'] if not pd.isna(last_row['RSI']) else 50)

    recent_df = df.tail(20)
    support = clean_val(recent_df['Low'].min())
    resistance = clean_val(recent_df['High'].max())

    entry_low = round(support, 2)
    entry_high = round(support + (price - support) * 0.35, 2)
    in_range = (entry_low <= price <= entry_high)

    rsi_status = "Sobrecomprada" if rsi > 70 else ("Sobrevendida" if rsi < 30 else "Neutra")

    return {
        'TF': period_name,
        'Precio': round(price, 2),
        'EMA_21_31': ema_21 > ema_31,
        'Price_150_200': (price > ema_150) and (price > ema_200),
        'RSI': round(rsi, 1),
        'RSI_Status': rsi_status,
        'Soporte': support,
        'Resistencia': resistance,
        'Entrada_Str': f"${entry_low} -${entry_high}",
        'In_Range': in_range
    }

def send_telegram_message(token, chat_id, text):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Error al enviar Telegram: {e}")

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    tickers = get_all_assets_tickers()
    print(f"Iniciando escaneo de {len(tickers)} activos...")
    
    alerts_sent = 0

    for ticker in tickers:
        try:
            df_1d = yf.download(ticker, period='2y', interval='1d', progress=False)
            df_1h = yf.download(ticker, period='60d', interval='1h', progress=False)

            if df_1d.empty or df_1h.empty:
                continue

            if isinstance(df_1d.columns, pd.MultiIndex):
                df_1d.columns = df_1d.columns.get_level_values(0)
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.get_level_values(0)

            df_4h = df_1h.resample('4h').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            analyses = [
                analyze_timeframe(df_1d, "1D"),
                analyze_timeframe(df_4h, "4H"),
                analyze_timeframe(df_1h, "1H")
            ]

            for res in analyses:
                if res:
                    # Condición estricta requerida
                    if res['EMA_21_31'] and res['Price_150_200']:
                        # Si además está en rango de entrada, mandamos alerta
                        if res['In_Range'] and token and chat_id:
                            msg = (
                                f"🚨 *¡ALERTA DE ENTRADA TÉCNICA!* 🚨\n\n"
                                f"📈 *Activo:* `{ticker}`\n"
                                f"⏱ *Temporalidad:* `{res['TF']}`\n"
                                f"💵 *Precio Actual:* `${res['Precio']:,}`\n"
                                f"🎯 *Zona de Entrada:* `{res['Entrada_Str']}`\n"
                                f"🛡 *Soporte:* `${res['Soporte']:,}`\n"
                                f"📊 *RSI:* `{res['RSI']}` ({res['RSI_Status']})\n\n"
                                f"_El precio está en zona óptima de pullback._"
                            )
                            send_telegram_message(token, chat_id, msg)
                            alerts_sent += 1
        except Exception:
            continue

    print(f"Escaneo finalizado. Alertas enviadas: {alerts_sent}")

if __name__ == "__main__":
    main()
