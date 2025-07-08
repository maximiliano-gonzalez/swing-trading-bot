import pandas as pd
import requests
import ta
import config
import yfinance as yf
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
from datetime import datetime

# === Parámetros configurables globales ===
RIESGO_PORTRADE = 1     # Porcentaje de capital arriesgado por operación
INVERSION_MINIMA = 10   # Mínimo a invertir por trade (en USD)
BENEFICIO_NETO_MIN = 2  # Beneficio neto estimado mínimo por trade (en USD)
COMISION_MIN = 1        # Comisión estimada de compra/venta (por trade, en USD)

def get_stock_data(ticker):
    print(f"[{ticker}] Consultando datos en Yahoo Finance...")
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty:
            print(f"[{ticker}] No se encontraron datos.")
            return None
        df = df.rename(columns={
            "Open": "o", "High": "h", "Low": "l", "Close": "c", "Adj Close": "adj_c", "Volume": "v"
        })
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"[{ticker}] Error al obtener datos: {e}")
        return None

def calculate_indicators(df):
    close = df["c"].squeeze()
    high = df["h"].squeeze()
    low = df["l"].squeeze()
    df["ema20"] = ta.trend.ema_indicator(close, window=20)
    df["ema50"] = ta.trend.ema_indicator(close, window=50)
    df["ema200"] = ta.trend.ema_indicator(close, window=200)
    df["rsi"] = ta.momentum.rsi(close, window=14)
    df["macd"] = ta.trend.macd(close)
    df["signal"] = ta.trend.macd_signal(close)
    df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    df["vol_ma20"] = df["v"].rolling(20).mean()
    return df

def check_trend(df):
    latest = df.iloc[-1]
    return latest["ema50"] > latest["ema200"]

def is_near_earnings(ticker):
    try:
        earnings = yf.Ticker(ticker).calendar
        if "Earnings Date" in earnings.index:
            earnings_date = earnings.loc["Earnings Date"].values[0]
            if pd.isna(earnings_date):
                return False
            if isinstance(earnings_date, (list, tuple, pd.Series)):
                earnings_date = earnings_date[0]
            if isinstance(earnings_date, pd.Timestamp):
                days_until = (earnings_date - pd.Timestamp.today()).days
                return 0 <= days_until <= 7
    except Exception as e:
        print(f"[{ticker}] ⚠️ Error al obtener earnings: {e}")
    return False

def get_last_pivot_support(df, lookback=30, left=3, right=3):
    lows = df['l'].tail(lookback)
    idxs = lows.index
    for i in range(left, len(lows)-right):
        pivot = True
        for j in range(1, left+1):
            if lows.iloc[i] >= lows.iloc[i-j]:
                pivot = False
        for j in range(1, right+1):
            if lows.iloc[i] >= lows.iloc[i+j]:
                pivot = False
        if pivot:
            last_idx = idxs[i]
            last_support = df.loc[last_idx, 'l']
            last_vol = df.loc[last_idx, 'v']
            return last_support, last_vol
    return None, None

def get_last_pivot_resistance(df, lookback=30, left=3, right=3):
    highs = df['h'].tail(lookback)
    idxs = highs.index
    for i in range(left, len(highs)-right):
        pivot = True
        for j in range(1, left+1):
            if highs.iloc[i] <= highs.iloc[i-j]:
                pivot = False
        for j in range(1, right+1):
            if highs.iloc[i] <= highs.iloc[i+j]:
                pivot = False
        if pivot:
            last_idx = idxs[i]
            last_res = df.loc[last_idx, 'h']
            last_vol = df.loc[last_idx, 'v']
            return last_res, last_vol
    return None, None

def check_conditions(df, ticker, capital):
    latest = df.iloc[-1]
    trend = latest["ema50"] > latest["ema200"]
    momentum = latest["ema20"] > latest["ema50"]
    price_above = latest["c"] > latest["ema20"] and latest["c"] > latest["ema50"] and latest["c"] > latest["ema200"]
    rsi_ok = 40 < latest["rsi"] < 70
    macd_ok = latest["macd"] > latest["signal"]

    comentario = []
    if trend: comentario.append("Tendencia alcista (EMA50>EMA200)")
    if momentum: comentario.append("Momentum positivo (EMA20>EMA50)")
    if price_above: comentario.append("Precio por encima de todas las EMAs")
    if latest["rsi"] > 60: comentario.append(f"RSI fuerte ({round(latest['rsi'],1)})")
    if macd_ok: comentario.append("MACD en compra")
    if latest["c"] > df["c"].rolling(20).max().iloc[-2]: comentario.append("Ruptura de máximos recientes")
    if latest["atr"] > df["atr"].mean(): comentario.append("ATR superior a la media (alta volatilidad)")
    comentario_str = "; ".join(comentario) or "Condiciones técnicas cumplidas"

    if trend and momentum and price_above and rsi_ok and macd_ok:
        print(f"[{ticker}] ✅ Señal detectada")
        entry = latest["c"]
        atr = latest["atr"]

        # --- Soporte/Resistencia por pivotes y volumen ---
        last_support, support_vol = get_last_pivot_support(df)
        last_resistance, resistance_vol = get_last_pivot_resistance(df)
        vol_ma20 = df['vol_ma20'].iloc[-1]
        support_ok = last_support is not None and support_vol is not None and support_vol >= vol_ma20
        resistance_ok = last_resistance is not None and resistance_vol is not None and resistance_vol >= vol_ma20

        # Determinación de SL y TP
        if support_ok:
            stop_loss = last_support * 0.995
            print(f"[{ticker}] SL por pivote: {round(stop_loss, 2)} (vol: {int(support_vol)}, MA20: {int(vol_ma20)})")
            comentario.append("SL por soporte pivote con volumen alto")
        else:
            stop_loss = entry - atr
            print(f"[{ticker}] SL por ATR: {round(stop_loss, 2)}")
            comentario.append("SL por ATR")

        if resistance_ok:
            take_profit = last_resistance * 0.995
            print(f"[{ticker}] TP por pivote: {round(take_profit, 2)} (vol: {int(resistance_vol)}, MA20: {int(vol_ma20)})")
            comentario.append("TP por resistencia pivote con volumen alto")
        else:
            take_profit = entry + (entry - stop_loss) * 1.5
            print(f"[{ticker}] TP clásico (RR 1.5x): {round(take_profit, 2)}")
            comentario.append("TP por RR clásico")

        # 🚨 Nuevas comprobaciones robustas
        if stop_loss >= entry:
            print(f"[{ticker}] ❌ SL mayor o igual al precio de entrada. Descartando trade.")
            return None
        if take_profit <= entry:
            print(f"[{ticker}] ❌ TP menor o igual al precio de entrada. Descartando trade.")
            return None

        # === Validar relación recompensa/riesgo (RR ratio) ===
        risk = entry - stop_loss
        reward = take_profit - entry
        rr_ratio = reward / risk
        if rr_ratio < 1.5:
            print(f"[{ticker}] ❌ RR ratio insuficiente: {rr_ratio:.2f}")
            return None

        # === Confirmación por volumen ===
        if latest["v"] < df["vol_ma20"].iloc[-1]:
            print(f"[{ticker}] ❌ Volumen bajo. Se descarta la señal.")
            return None

        risk_usd = capital * (RIESGO_PORTRADE / 100)
        risk_per_share = entry - stop_loss
        if risk_per_share <= 0:
            print(f"[{ticker}] ⚠️ Riesgo por acción inválido (<=0).")
            return None
        position_size = risk_usd / risk_per_share
        position_usd = entry * position_size

        beneficio_estimado = (take_profit - entry) * position_size - COMISION_MIN
        perdida_estimada = (entry - stop_loss) * position_size + COMISION_MIN

        if is_near_earnings(ticker):
            print(f"[{ticker}] ⚠️ Earnings en menos de 7 días. Se descarta la señal.")
            return None
        if position_usd < INVERSION_MINIMA or beneficio_estimado < BENEFICIO_NETO_MIN:
            print(f"[{ticker}] ❌ No cumple mínimos de inversión o beneficio esperado.")
            return None

        return {
            "ticker": ticker,
            "price": round(latest['c'], 2),
            "ema20": round(latest['ema20'], 2),
            "ema50": round(latest['ema50'], 2),
            "ema200": round(latest['ema200'], 2),
            "rsi": round(latest['rsi'], 2),
            "macd": round(latest['macd'], 2),
            "atr": round(latest['atr'], 2),
            "time": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
            "comentario": "; ".join(comentario),
            "entry": round(entry, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "position_size": round(position_size, 2),
            "position_usd": round(position_usd, 2),
            "beneficio_neto_estimado": round(beneficio_estimado, 2),
            "perdida_estimada": round(perdida_estimada, 2),
            "rr_ratio": round(rr_ratio, 2),
        }
    else:
        print(f"[{ticker}] ❌ No hay señal")
        return None

def send_webhook(signal):
    response = requests.post(config.WEBHOOK_URL, json=signal)
    print(f"Webhook enviado: status {response.status_code}")

def get_capital_from_sheets():
    print("🔎 Consultando operaciones para actualizar capital...")
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('service_account.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open("Operaciones Bolsa").sheet1
    records = sheet.get_all_records()

    capital_base = 200.0
    capital_actual = capital_base

    for op in records:
        cierre = str(op.get("Cierre?", "")).strip().lower()
        if not cierre or cierre in ["abierta", ""]:
            continue
        if "tp" in cierre:
            capital_actual += float(op.get("Beneficio estimado", 0))
        elif "sl" in cierre:
            capital_actual -= float(op.get("Pérdida estimada", 0))

    with open("capital.txt", "w") as f:
        f.write(str(round(capital_actual, 2)))

    print(f"✅ Capital actualizado: ${capital_actual:.2f}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_file = "capital_history.csv"
    if not os.path.exists(history_file):
        with open(history_file, "w") as f:
            f.write("Fecha,Capital\n")
    with open(history_file, "a") as f:
        f.write(f"{now},{round(capital_actual,2)}\n")

    return capital_actual

def main():
    capital = get_capital_from_sheets()

    assets = pd.read_csv("assets.csv")
    assets.columns = assets.columns.str.strip().str.lower().str.replace(" ", "_")
    ticker_col = "ticker"
    etf_col = None
    for col in assets.columns:
        if "etf" in col and "sector" in col:
            etf_col = col
            break
    if not etf_col:
        raise Exception("No se encontró la columna ETF sectorial en assets.csv")

    for index, row in assets.iterrows():
        ticker = row[ticker_col]
        sector_etf = row[etf_col]
        if pd.isna(sector_etf) or sector_etf == "" or ticker == sector_etf:
            print(f"[{ticker}] Es ETF sectorial o no tiene ETF asociado. Se ignora para señales.")
            continue

        print(f"\nAnalizando {ticker} (ETF: {sector_etf})...")
        df = get_stock_data(ticker)
        if df is None or df.empty:
            continue
        df = calculate_indicators(df)
        expected_cols = ["ema20", "ema50", "ema200", "rsi", "macd", "signal", "atr"]
        if not set(expected_cols).issubset(df.columns):
            continue
        df = df.dropna(subset=expected_cols)
        if df.empty:
            continue
        signal = check_conditions(df, ticker, capital)
        if not signal:
            continue

        df_etf = get_stock_data(sector_etf)
        if df_etf is None or df_etf.empty:
            print(f"[{sector_etf}] ❌ No se pudo obtener información del ETF sectorial.")
            continue
        df_etf = calculate_indicators(df_etf)
        if not set(expected_cols).issubset(df_etf.columns):
            continue
        df_etf = df_etf.dropna(subset=expected_cols)
        if df_etf.empty:
            continue
        if check_trend(df_etf):
            print(f"[{ticker}] y [{sector_etf}] ✅ Ambos en tendencia alcista. Enviando señal.")
            send_webhook(signal)
        else:
            print(f"[{ticker}] ✅ Señal en acción, pero [{sector_etf}] no está alcista. NO se envía señal.")

if __name__ == "__main__":
    main()
