import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ================= CONFIGURACIÓN DE TELEGRAM =================
# Las credenciales se leen automáticamente de los Secrets de GitHub Actions
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_telegram_message(message):
    """Envía una notificación al chat de Telegram configurado."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Advertencia: Credenciales de Telegram no configuradas.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Error al enviar Telegram: {response.text}")
    except Exception as e:
        print(f"Error de conexión con Telegram: {e}")

# ================= LISTA COMPLETA DE ACTIVOS =================
def get_all_assets_tickers():
    """Retorna la lista completa del NASDAQ 100 + Criptos + XAUT."""
    nasdaq_tickers = [
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD',
        'AMGN', 'AMZN', 'ANSS', 'APP', 'ARM', 'ASML', 'AVGO', 'AXON', 'AZN', 'BIIB',
        'BKNG', 'BKR', 'CCEP', 'CDNS', 'CDW', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT',
        'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTSH', 'DASH', 'DDOG', 'DLTR', 'DXCM',
        'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GFS', 'GOOG', 'GOOGL', 'HON', 'IDEXX',
        'ILMN', 'INTC', 'INTU', 'ISRG', 'KDP', 'KHC', 'KLAC', 'LIN', 'LRCX', 'LULU',
        'MAR', 'MCHP', 'MDLZ', 'MELI', 'META', 'MNST', 'MRNA', 'MSFT', 'MU', 'NFLX',
        'NVDA', 'NXPI', 'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP',
        'QCOM', 'REGN', 'ROP', 'ROST', 'SBUX', 'SMR', 'SNPS', 'TEAM', 'TMUS', 'TSLA',
        'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS'
    ]
    
    crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'XAUT-USD'
    ]
    
    return nasdaq_tickers + crypto_tickers

# ================= CÁLCULO TÉCNICO =================
def calculate_indicators(df):
    """Calcula EMAs, RSI y Soportes/Resistencias."""
    if df is None or len(df) < 200:
        return None
    
    # EMAs
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_31'] = df['Close'].ewm(span=31, adjust=False).mean()
    df['EMA_150'] = df['Close'].ewm(span=150, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # RSI (14 períodos)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Soporte dinámico (mínimo de las últimas 20 velas)
    df['Support'] = df['Low'].rolling(window=20).min()
    
    return df

def analyze_asset(ticker):
    """Descarga datos y evalúa las condiciones en temporalidades 1D, 4H y 1H."""
    timeframes = {'1D': '1d', '4H': '60m', '1H': '60m'} # Nota: yfinance maneja 60m para intradía
    
    # Usaremos un diccionario de temporalidades ajustadas para yfinance
    tf_configs = {
        '1D': {'interval': '1d', 'period': '1y'},
        '4H': {'interval': '1h', 'period': '60d'}, # Simulación o datos por hora
        '1H': {'interval': '60m', 'period': '30d'}
    }

    for tf, config in tf_configs.items():
        try:
            df = yf.download(ticker, interval=config['interval'], period=config['period'], progress=False)
            if df.empty or len(df) < 200:
                continue
                
            # Limpiar multiindex si yfinance lo devuelve
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = calculate_indicators(df)
            if df is None:
                continue
                
            last = df.iloc[-1]
            price = last['Close']
            ema21 = last['EMA_21']
            ema31 = last['EMA_31']
            ema150 = last['EMA_150']
            ema200 = last['EMA_200']
            rsi = last['RSI']
            support = last['Support']
            
            # --- CONDICIONES TÉCNICAS ---
            # 1. Tendencia Corto Plazo: EMA 21 > EMA 31
            cond_ema_short = ema21 > ema31
            # 2. Tendencia Largo Plazo: Precio > EMA 150 y EMA 200
            cond_ema_long = (price > ema150) and (price > ema200)
            
            # 3. Zona de Entrada Óptima (Pullback al soporte con un margen del 35% hacia arriba)
            upper_entry_zone = support + ((price - support) * 0.35)
            cond_entry_zone = (price >= support) and (price <= upper_entry_zone)
            
            if cond_ema_short and cond_ema_long and cond_entry_zone:
                # Determinar estado del RSI
                rsi_status = "Neutra"
                if rsi > 70:
                    rsi_status = "Sobrecomprada"
                elif rsi < 30:
                    rsi_status = "Sobrevendida"
                
                # Formatear mensaje de alerta
                msg = (
                    f"🚨 *¡ALERTA DE ENTRADA TÉCNICA!* 🚨\n\n"
                    f"📈 *Activo:* `{ticker}`\n"
                    f"⏱ *Temporalidad:* `{tf}`\n"
                    f"💵 *Precio Actual:* `${price:,.2f}`\n"
                    f"🎯 *Soporte Clave:* `${support:,.2f}`\n"
                    f"📊 *RSI:* `{rsi:.1f}` ({rsi_status})\n\n"
                    f"_El precio está testeando la zona óptima de pullback._"
                )
                send_telegram_message(msg)
                print(f"[ALERTA ENVIADA] {ticker} en {tf}")

        except Exception as e:
            print(f"Error analizando {ticker} en {tf}: {e}")

def main():
    print("Iniciando escaneo de mercado...")
    tickers = get_all_assets_tickers()
    print(f"Total de activos a escanear: {len(tickers)}")
    
    for ticker in tickers:
        print(f"Escaneando {ticker}...")
        analyze_asset(ticker)
        
    print("Escaneo finalizado.")

if __name__ == "__main__":
    main()
