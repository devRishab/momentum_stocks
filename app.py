import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Nifty 500 Momentum Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS ---
CRORE = 10**7
THRESH_1 = 10000 * CRORE   # 10,000 Crore
THRESH_2 = 20000 * CRORE   # 20,000 Crore
CHUNK_SIZE = 50
MAX_WORKERS = 10

# --- HELPER FUNCTIONS (CACHED) ---
@st.cache_data(ttl=3600)  # Cache list for 1 hour
def fetch_live_nifty500():
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return [f"{sym.strip()}.NS" for sym in df['Symbol']]
    except Exception as e:
        st.error(f"Error fetching live universe: {e}")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "SUZLON.NS", "ZOMATO.NS"]

def get_market_cap(ticker):
    try:
        t = yf.Ticker(ticker)
        mcap = t.fast_info.get('marketCap', t.fast_info.get('market_cap', None))
        return ticker, mcap
    except:
        return ticker, None

@st.cache_data(ttl=1800)  # Cache calculations for 30 mins to avoid re-running on UI clicks
def run_momentum_pipeline(tickers, lookback_months):
    all_returns = {}
    all_monthly_series = {}
    
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(months=lookback_months)
    min_trading_days = max(5, lookback_months * 15)
    
    # Setup placeholder for progress reporting within the app
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tickers = len(tickers)
    
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        status_text.text(f"Scanning market data: Processing stocks {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        progress_bar.progress(min(i / total_tickers, 1.0))
        
        try:
            data = yf.download(chunk, start=start_date, end=end_date, interval="1d", group_by='ticker', progress=False)
            for ticker in chunk:
                try:
                    if ticker in data.columns:
                        df_ticker = data[ticker].dropna(subset=['Close'])
                        if len(df_ticker) < min_trading_days: 
                            continue 
                        
                        start_val = df_ticker['Close'].iloc[0]
                        end_val = df_ticker['Close'].iloc[-1]
                        horizon_return = ((end_val - start_val) / start_val) * 100
                        
                        monthly = df_ticker['Close'].resample('ME').last().pct_change().dropna() * 100
                        all_returns[ticker] = horizon_return
                        all_monthly_series[ticker] = monthly
                except:
                    continue
        except:
            continue
        time.sleep(0.2)
        
    progress_bar.progress(0.9)
    status_text.text("Verifying current market capitalizations...")
    
    active_tickers = list(all_returns.keys())
    market_caps = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_market_cap, tk): tk for tk in active_tickers}
        for future in as_completed(futures):
            tk, mcap = future.result()
            if mcap and mcap > 0:
                market_caps[tk] = mcap
                
    progress_bar.empty()
    status_text.empty()
    
    consolidated = []
    for tk in market_caps:
        consolidated.append({
            'Ticker': tk,
            'MarketCap': market_caps[tk],
            'Return_Horizon': all_returns[tk],
            'Monthly_Returns': all_monthly_series[tk]
        })
        
    return pd.DataFrame(consolidated)

def render_heatmap(df, title):
    if df.empty:
        st.warning(f"No stocks found meeting the criteria for {title}.")
        return
    
    matrix_data = pd.DataFrame({row['Ticker']: row['Monthly_Returns'] for _, row in df.iterrows()}).T
    matrix_data.columns = [col.strftime('%b %y') for col in matrix_data.columns]
    
    fig, ax = plt.subplots(figsize=(12, min(8, len(df) * 0.4 + 2)))
    sns.heatmap(matrix_data, annot=True, cmap='RdYlGn', center=0, fmt=".1f", linewidths=0.4, ax=ax, cbar_kws={'label': 'Return %'})
    ax.set_title(f"{title} - Monthly Momentum Breakdown", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Timeline Track", fontsize=10)
    ax.set_ylabel("Ticker Symbol", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🛠️ Configuration Panel")
st.sidebar.markdown("Fine-tune your momentum tracking filters below:")

user_months = st.sidebar.slider(
    "Lookback Horizon (Months)", 
    min_value=1, 
    max_value=24, 
    value=3, 
    step=1,
    help="The time window used to calculate total momentum returns."
)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Pro-Tip**: Changing lookback values triggers a fresh analysis. Switching result tabs afterwards is instantaneous due to caching.")

# --- MAIN DASHBOARD INTERFACE ---
st.title("📈 Nifty 500 Quantitative Momentum Dashboard")
st.markdown("This application scans the live **Nifty 500 universe**, automatically buckets equities by market cap, isolates the **Top 20 absolute momentum leaders**, and maps out their monthly performance footprint.")

# Run calculations
universe = fetch_live_nifty500()

with st.spinner("Analyzing market dynamics..."):
    master_data = run_momentum_pipeline(universe, lookback_months=user_months)

if not master_data.empty:
    # Segments
    small_cap = master_data[master_data['MarketCap'] <= THRESH_1]
    mid_cap = master_data[(master_data['MarketCap'] > THRESH_1) & (master_data['MarketCap'] <= THRESH_2)]
    large_cap = master_data[master_data['MarketCap'] > THRESH_2]
    
    # Layout tabs
    tab1, tab2, tab3 = st.tabs(["🏛️ Large Cap (> 20k Cr)", "🏢 Mid Cap (10k - 20k Cr)", "🌱 Small Cap (<= 10k Cr)"])
    
    categories = [
        (tab1, "Large Cap (> 20k Crore)", large_cap),
        (tab2, "Mid Cap (10k to 20k Crore)", mid_cap),
        (tab3, "Small Cap (<= 10k Crore)", small_cap)
    ]
    
    for tab, name, segment_df in categories:
        with tab:
            top_20 = segment_df.sort_values(by='Return_Horizon', ascending=False).head(20)
            
            if not top_20.empty:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Leaderboard")
                    # Clean presentation of the dataframe
                    display_df = top_20[['Ticker', 'Return_Horizon']].copy()
                    display_df['Market Cap (Cr)'] = (top_20['MarketCap'] / CRORE).round(2)
                    display_df = display_df.rename(columns={'Return_Horizon': f'Return ({user_months}M) %'})
                    display_df = display_df.reset_index(drop=True)
                    display_df.index += 1
                    
                    st.dataframe(
                        display_df, 
                        use_container_width=True,
                        column_config={
                            f"Return ({user_months}M) %": st.column_config.NumberColumn(format="%.2f%%"),
                            "Market Cap (Cr)": st.column_config.NumberColumn(format="₹%,.2f")
                        }
                    )
                
                with col2:
                    st.subheader("Performance Consistency Heatmap")
                    render_heatmap(top_20, name)
            else:
                st.info(f"No active Nifty 500 components currently match the market cap boundaries for {name}.")
else:
    st.error("Failed to recover active stock streams from the index network interface.")
