import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from main import calculate_indicators, check_conditions  # <-- importa tus funciones

RIESGO_PORTRADE = 1
INVERSION_MINIMA = 10
BENEFICIO_NETO_MIN = 2
COMISION_MIN = 1

INITIAL_CAPITAL = 500
START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # Últimos 2 años

def get_stock_data(ticker):
    df = yf.download(ticker, start=START_DATE, interval="1d", progress=False)
    if df.empty:
        return None
    df = df.rename(columns={
        "Open": "o", "High": "h", "Low": "l", "Close": "c", "Adj Close": "adj_c", "Volume": "v"
    })
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def run_backtest():
    assets = pd.read_csv("assets.csv")
    assets.columns = assets.columns.str.strip().str.lower().str.replace(" ", "_")
    ticker_col = "ticker"
    etf_col = [col for col in assets.columns if "etf" in col and "sector" in col][0]
    capital = INITIAL_CAPITAL
    peak = capital
    max_drawdown = 0
    wins = 0
    losses = 0
    trades = []
    
    for idx, row in assets.iterrows():
        ticker = row[ticker_col]
        sector_etf = row[etf_col]
        if pd.isna(sector_etf) or sector_etf == "" or ticker == sector_etf:
            continue
        print(f"\nBacktesteando {ticker}...")
        df = get_stock_data(ticker)
        if df is None or df.empty or len(df) < 60:
            continue
        df = calculate_indicators(df)
        expected_cols = ["ema20", "ema50", "ema200", "rsi", "macd", "signal", "atr"]
        if not set(expected_cols).issubset(df.columns):
            continue
        df = df.dropna(subset=expected_cols)
        if df.empty:
            continue

        # --- Recorre día a día el histórico ---
        for i in range(60, len(df)):
            df_slice = df.iloc[:i+1].copy()
            signal = check_conditions(df_slice, ticker, capital)
            if signal:
                entry_date = df_slice.index[-1]
                entry_price = signal['entry']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                shares = signal['position_size']
                # Simula cierre en los siguientes 10 días o hasta el final
                result = None
                for j in range(i+1, min(i+11, len(df))):
                    day = df.iloc[j]
                    if day['l'] <= stop_loss:
                        exit_date = df.index[j]
                        exit_price = stop_loss
                        result = 'SL'
                        break
                    if day['h'] >= take_profit:
                        exit_date = df.index[j]
                        exit_price = take_profit
                        result = 'TP'
                        break
                if not result:
                    exit_date = df.index[min(i+10, len(df)-1)]
                    exit_price = df.iloc[min(i+10, len(df)-1)]['c']
                    result = 'EXP'
                # Ganancia/pérdida neta
                if result == 'TP':
                    profit = (take_profit - entry_price) * shares - COMISION_MIN
                    wins += 1
                elif result == 'SL':
                    profit = (entry_price - stop_loss) * shares - COMISION_MIN
                    losses += 1
                else:
                    profit = (exit_price - entry_price) * shares - COMISION_MIN
                    # No cuenta como win/loss pero suma al capital
                capital += profit
                peak = max(peak, capital)
                max_drawdown = max(max_drawdown, (peak - capital))
                trades.append({
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry': entry_price,
                    'exit': exit_price,
                    'result': result,
                    'profit': profit,
                    'capital': capital
                })

    df_trades = pd.DataFrame(trades)
    df_trades.to_csv('backtest_results.csv', index=False)

    total_trades = wins + losses
    winrate = (wins / total_trades * 100) if total_trades > 0 else 0
    avg_profit = df_trades['profit'].mean() if not df_trades.empty else 0

    print(f"\nCapital final: {capital:.2f} €")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Ganancia promedio por trade: {avg_profit:.2f} €")
    print(f"Máximo drawdown: {max_drawdown:.2f} €")
    print("Resultados guardados en backtest_results.csv")

    return df_trades

if __name__ == "__main__":
    df_trades = run_backtest()

    import matplotlib.pyplot as plt

    if not df_trades.empty:
        # Equity curve diaria realista
        df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
        df_trades = df_trades.sort_values('exit_date')

        start = df_trades['exit_date'].min()
        end = df_trades['exit_date'].max()
        dates = pd.date_range(start, end)

        capital = [INITIAL_CAPITAL]
        date_index = [dates[0]]

        current_cap = INITIAL_CAPITAL

        for d in dates[1:]:
            # Suma todos los profits cerrados en este día (pueden ser varios)
            daily_profit = df_trades[df_trades['exit_date'] == d]['profit'].sum()
            current_cap += daily_profit
            capital.append(current_cap)
            date_index.append(d)

        plt.figure(figsize=(12,6))
        plt.plot(date_index, capital, marker='', linestyle='-')
        plt.title('Equity Curve Realista (capital día a día)')
        plt.xlabel('Fecha')
        plt.ylabel('Capital (€)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig("equity_curve_real.png")
        plt.show()
    else:
        print("No se generaron operaciones para graficar.")


df = pd.read_csv('backtest_results.csv')

# Operación ganadora más grande
max_gain = df.loc[df['profit'].idxmax()]
print("🏆 Operación ganadora más grande:")
print(max_gain)
print()

# Mayor pérdida (más negativa)
max_loss = df.loc[df['profit'].idxmin()]
print("💥 Mayor pérdida registrada:")
print(max_loss)

