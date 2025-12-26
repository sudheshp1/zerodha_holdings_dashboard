import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from urllib.parse import urlparse, parse_qs
import plotly.express as px
from datetime import datetime, time
import yfinance as yf
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="Kite Portfolio Dashboard", page_icon="💰", layout="wide")

st.title("💰 Zerodha Kite Full Portfolio Dashboard")

# --- Updated Market Benchmarks (Multi-Index Fixed) ---
@st.cache_data(ttl=600)
def get_benchmarks():
    # Updated Ticker List for Yahoo Finance
    tickers = {
        "Nifty 50": "^NSEI", 
        "Nifty Next 50": "^NSMIDCP", 
        "Nifty Midcap 150": "NIFTYMIDCAP150.NS"
        #"Sensex": "^BSESN"
    }
    results = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty and len(df) >= 2:
                # Handle potential Multi-Index columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close_series = df['Close'][ticker]
                else:
                    close_series = df['Close']
                
                # Convert to float to avoid "unsupported format string" error
                ltp = float(close_series.iloc[-1])
                prev_close = float(close_series.iloc[-2])
                results[name] = (ltp, prev_close)
        except:
            continue
    return results

try:
    benchmarks = get_benchmarks()
    if benchmarks:
        # Using 4 columns for the 4 indices
        cols = st.columns(4)
        for i, (name, (ltp, prev_close)) in enumerate(benchmarks.items()):
            change = ltp - prev_close
            pct_change = (change / prev_close) * 100
            
            cols[i].metric(
                label=name, 
                value=f"{ltp:,.2f}", 
                delta=f"{change:+.2f} ({pct_change:+.2f}%)"
            )
except Exception as e:
    st.error(f"Could not load benchmarks: {e}")

st.divider()


# --- Step 1: Configuration ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("API Key", value="e53bh9l0p0haf9wl")
    api_secret = st.text_input("API Secret", value="4k4ft0rn9xb23dxeqt9nsb6gtonrsg3z", type="password")
    
    st.divider()
    
    if api_key:
        kite = KiteConnect(api_key=api_key)
        st.write("1. Authorize App:")
        if st.button("🚀 Get Login URL"):
            st.link_button("Login to Kite", kite.login_url())
            
    redirect_url = st.text_input("2. Paste Redirect URL here:")

    # --- NEW: What-If Analysis Slider ---
    st.divider()
    st.markdown("### 🧪 What-If Analysis")
    simulation_pct = st.slider(
        "Simulate Market Change (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=1,
        help="See how your portfolio value changes if the market moves by this %"
    )
    if simulation_pct != 0:
        st.sidebar.warning(f"Simulating a {simulation_pct}% {'gain' if simulation_pct > 0 else 'drop'}")

# --- Helper Functions ---
def color_pnl(val):
    color = 'green' if val >= 0 else 'red'
    return f'color: {color}'

def color_status(val):
    color = 'green' if val == 'ACTIVE' else 'orange' if val == 'PAUSED' else 'red'
    return f'color: {color}; font-weight: bold'

# --- Step 2: Main Logic ---
if redirect_url and api_secret:
    try:
        parsed_url = urlparse(redirect_url)
        request_token = parse_qs(parsed_url.query).get("request_token", [None])[0]

        if request_token:
            @st.cache_resource
            def get_kite_session(token):
                data = kite.generate_session(token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                return kite

            kite_session = get_kite_session(request_token)
            
            with st.spinner("Fetching Holdings..."):
                user_profile = kite_session.profile()
                holdings = kite_session.holdings()
                mf_holdings = kite_session.mf_holdings()
                mf_sips = kite_session.mf_sips() 

                st.markdown("### 👤 Account Details")
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                with p_col1:
                    st.caption("User ID")
                    st.subheader(user_profile['user_id'])
                with p_col2:
                    st.caption("Username")
                    st.subheader(user_profile['user_name'])
                with p_col3:
                    st.caption("Broker")
                    st.subheader(user_profile['broker'])
                with p_col4:
                    st.caption("Email ID")
                    st.write(user_profile['email'])

                st.divider()

                # --- STEP A: PREPARE DATAFRAMES & SIMULATION ---
                multiplier = 1 + (simulation_pct / 100)
                
                df_eq = pd.DataFrame(holdings) if holdings else pd.DataFrame()
                df_mf = pd.DataFrame(mf_holdings) if mf_holdings else pd.DataFrame()

                if not df_eq.empty:
                    df_eq['invested_value'] = df_eq['quantity'] * df_eq['average_price']
                    # Apply simulation to current value
                    df_eq['current_value'] = (df_eq['quantity'] * df_eq['last_price']) * multiplier

                if not df_mf.empty:
                    df_mf['invested_value'] = df_mf['quantity'] * df_mf['average_price']
                    # Apply simulation to current value
                    df_mf['current_value'] = (df_mf['quantity'] * df_mf['last_price']) * multiplier

                # --- STEP B: VISUALIZATIONS ---
                st.markdown("### 📈 Portfolio Composition")
                
                total_eq_val = df_eq['current_value'].sum() if not df_eq.empty else 0
                total_mf_val = df_mf['current_value'].sum() if not df_mf.empty else 0

                viz_col1, viz_col2 = st.columns(2)
                with viz_col1:
                    fig_asset = px.pie(
                        names=["Equity", "Mutual Funds"], 
                        values=[total_eq_val, total_mf_val], 
                        hole=0.5, title="Asset Allocation (Simulated)" if simulation_pct != 0 else "Asset Allocation",
                        color_discrete_sequence=['#00d2ff', '#3a7bd5']
                    )
                    st.plotly_chart(fig_asset, use_container_width=True)

                with viz_col2:
                    view_option = st.radio(
                        label="View Top 10 Holdings for:",
                        options=["All", "Equity", "Mutual Funds"],
                        horizontal=True,
                        key="top_10_selector"
                    )

                    if view_option == "Equity":
                        df_top_10 = df_eq[['tradingsymbol', 'current_value']].rename(columns={'tradingsymbol': 'Name'}) if not df_eq.empty else pd.DataFrame()
                    elif view_option == "Mutual Funds":
                        df_top_10 = df_mf[['fund', 'current_value']].rename(columns={'fund': 'Name'}) if not df_mf.empty else pd.DataFrame()
                    else:
                        df_eq_sub = df_eq[['tradingsymbol', 'current_value']].rename(columns={'tradingsymbol': 'Name'}) if not df_eq.empty else pd.DataFrame()
                        df_mf_sub = df_mf[['fund', 'current_value']].rename(columns={'fund': 'Name'}) if not df_mf.empty else pd.DataFrame()
                        df_top_10 = pd.concat([df_eq_sub, df_mf_sub])

                    if not df_top_10.empty:
                        df_top_10 = df_top_10.sort_values(by='current_value', ascending=False).head(10)
                        fig_tree = px.treemap(
                            df_top_10, 
                            path=[px.Constant("Portfolio"), 'Name'], 
                            values='current_value', 
                            title=f"Top 10 {view_option} Holdings",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_tree.update_traces(
                            texttemplate="<b>%{label}</b><br>₹%{value:,.0f}", 
                            textfont_size=20,
                            hovertemplate="<b>%{label}</b><br>Value: ₹%{value:,.2f}<extra></extra>"
                        )
                        fig_tree.update_layout(margin=dict(t=30, l=0, r=0, b=0))
                        st.plotly_chart(fig_tree, use_container_width=True)
                    else:
                        st.info(f"No {view_option} data available.")

                st.divider()

            # Create Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Equity Holdings", "🏦 Mutual Funds", "📅 MF SIPs"])

            with tab1:
                if not df_eq.empty:
                    # Calculations for Tab 1 (already simulated in Step A)
                    df_eq['pnl'] = df_eq['current_value'] - df_eq['invested_value']
                    df_eq['pnl_pct'] = (df_eq['pnl'] / df_eq['invested_value']) * 100
                    df_eq = df_eq.sort_values(by='current_value', ascending=False).reset_index(drop=True)
                    df_eq.index = df_eq.index + 1
                    
                    total_inv_eq = df_eq['invested_value'].sum()
                    current_val_eq = df_eq['current_value'].sum()
                    total_pnl_eq = current_val_eq - total_inv_eq
                    total_pnl_pct = (total_pnl_eq / total_inv_eq) * 100 if total_inv_eq != 0 else 0
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Equity Investment", f"₹{total_inv_eq:,.2f}")
                    c2.metric("Current Value" if simulation_pct == 0 else "Simulated Value", f"₹{current_val_eq:,.2f}")
                    c3.metric("Total P&L", f"₹{total_pnl_eq:,.2f}", delta=f"{total_pnl_pct:.2f}%")

                    disp_eq = df_eq[['tradingsymbol', 'quantity', 'average_price', 'invested_value', 'last_price', 'current_value', 'pnl', 'pnl_pct']].copy()
                    disp_eq.columns = ['Symbol', 'Qty', 'Avg. Price', 'Invested', 'LTP', 'Current Value', 'P&L', 'P&L %']
                    
                    st.dataframe(
                        disp_eq.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
                        .format({
                            'Avg. Price': '₹{:.2f}', 'Invested': '₹{:.2f}', 
                            'LTP': '₹{:.2f}', 'Current Value': '₹{:.2f}', 
                            'P&L': '₹{:.2f}', 'P&L %': '{:.2f}%'
                        }), use_container_width=True
                    )
                else:
                    st.info("No Equity holdings found.")

            with tab2:
                if not df_mf.empty:
                    # Calculations for Tab 2 (already simulated in Step A)
                    df_mf['pnl'] = df_mf['current_value'] - df_mf['invested_value']
                    df_mf['pnl_pct'] = (df_mf['pnl'] / df_mf['invested_value']) * 100
                    df_mf = df_mf.sort_values(by='current_value', ascending=False).reset_index(drop=True)
                    df_mf.index = df_mf.index + 1
                    
                    total_inv_mf = df_mf['invested_value'].sum()
                    current_val_mf = df_mf['current_value'].sum()
                    total_pnl_mf = current_val_mf - total_inv_mf
                    total_pnl_pct_mf = (total_pnl_mf / total_inv_mf) * 100 if total_inv_mf != 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("MF Investment", f"₹{total_inv_mf:,.2f}")
                    m2.metric("Current Value" if simulation_pct == 0 else "Simulated Value", f"₹{current_val_mf:,.2f}")
                    m3.metric("MF P&L", f"₹{total_pnl_mf:,.2f}", delta=f"{total_pnl_pct_mf:.2f}%")

                    disp_mf = df_mf[['fund', 'quantity', 'average_price', 'invested_value', 'last_price', 'current_value', 'pnl', 'pnl_pct']].copy()
                    disp_mf.columns = ['Fund Name', 'Units', 'Avg. NAV', 'Invested', 'Current NAV', 'Current Value', 'P&L', 'P&L %']
                    
                    st.dataframe(
                        disp_mf.style.applymap(color_pnl, subset=['P&L', 'P&L %'])
                        .format({
                            'Units': '{:.2f}', 'Avg. NAV': '₹{:.2f}',
                            'Invested': '₹{:.2f}', 'Current NAV': '₹{:.2f}',
                            'Current Value': '₹{:.2f}', 'P&L': '₹{:.2f}', 'P&L %': '{:.2f}%'
                        }), use_container_width=True
                    )
                else:
                    st.info("No Mutual Fund holdings found.")
            
            with tab3:
                if mf_sips:
                    df_sip = pd.DataFrame(mf_sips)
                    df_sip['instalment_day'] = pd.to_numeric(df_sip['instalment_day'])
                    active_sips = df_sip[df_sip['status'] == 'ACTIVE'].copy()
                    
                    total_sip_monthly = active_sips['instalment_amount'].sum()
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Monthly Commitment", f"₹{total_sip_monthly:,.2f}")
                    s2.metric("Active SIPs", len(active_sips))
                    
                    today_day = datetime.now().day
                    remaining_cash = active_sips[active_sips['instalment_day'] >= today_day]['instalment_amount'].sum()
                    s3.metric("Remaining This Month", f"₹{remaining_cash:,.2f}")

                    st.divider()

                    cash_flow_df = active_sips.groupby('instalment_day')['instalment_amount'].sum().reset_index()
                    fig_sip = px.bar(
                        cash_flow_df, x='instalment_day', y='instalment_amount',
                        title="Monthly SIP Cash Outflow Schedule",
                        labels={'instalment_day': 'Day of Month', 'instalment_amount': 'Amount (₹)'},
                        color_discrete_sequence=['#FF4B4B']
                    )
                    fig_sip.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1))
                    st.plotly_chart(fig_sip, use_container_width=True)

                    st.subheader("📋 Complete SIP Schedule")
                    disp_sip = df_sip.sort_values(by='instalment_day').reset_index(drop=True)
                    disp_sip.index = disp_sip.index + 1
                    disp_sip = disp_sip[['fund', 'status', 'instalment_amount', 'instalment_day', 'next_instalment']].copy()
                    disp_sip.columns = ['Fund Name', 'Status', 'Amount', 'Day', 'Next Date']
                    
                    st.dataframe(
                        disp_sip.style.applymap(color_status, subset=['Status'])
                        .format({'Amount': '₹{:.2f}'}), use_container_width=True
                    )
                else:
                    st.info("No Mutual Fund SIPs found.")

            # --- STEP D: HISTORICAL PERFORMANCE & FUNDAMENTALS ---
            st.divider()
            st.markdown("### 📊 Stock Analysis & Fundamentals")

            if not df_eq.empty:
                # 1. Selection Header
                input_col1, input_col2 = st.columns([2, 1])
                with input_col1:
                    stock_list = sorted(df_eq['tradingsymbol'].unique().tolist())
                    selected_stock = st.selectbox("Select stock to analyze:", stock_list)
                with input_col2:
                    time_period = st.select_slider("Time Period:", 
                                                options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], 
                                                value="1y")

                # 2. Fetch Data from yFinance
                yf_symbol = f"{selected_stock}.NS"
                ticker_obj = yf.Ticker(yf_symbol)
                
                with st.spinner("Fetching Market Data..."):
                    info = ticker_obj.info
                    
                    # 3. KPI Ribbon (Corrected to remove duplicate Market Cap)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sector", info.get('sector', 'N/A'), help=f"Industry: {info.get('industry', 'N/A')}")
                    
                    mcap = info.get('marketCap')
                    mcap_val = f"₹{mcap/10**7:,.0f} Cr" if mcap else "N/A"
                    m2.metric("Market Cap", mcap_val)
                    
                    pe = info.get('trailingPE')
                    m3.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
                    
                    div = info.get('dividendYield')
                    m4.metric("Div. Yield", f"{div:.2f}%" if div else "0.00%")

                # 6. Historical Line Chart
                st.markdown("---")
                hist_data = yf.download(yf_symbol, period=time_period, interval="1d", progress=False)
                if not hist_data.empty:
                    if isinstance(hist_data.columns, pd.MultiIndex):
                        hist_data.columns = hist_data.columns.get_level_values(0)
                    
                    fig_hist = px.line(hist_data, y='Close', 
                                    title=f"{selected_stock} - {time_period.upper()} Price Trend", 
                                    template="plotly_dark")
                    
                    avg_buy = float(df_eq[df_eq['tradingsymbol'] == selected_stock]['average_price'].iloc[0])
                    fig_hist.add_hline(y=avg_buy, line_dash="dash", line_color="#FFA500", 
                                    annotation_text=f"Your Avg: ₹{avg_buy:.2f}")
                    
                    fig_hist.update_layout(hovermode="x unified", margin=dict(t=50, l=0, r=0, b=0))
                    st.plotly_chart(fig_hist, use_container_width=True)

            # --- STEP E: FINANCIAL GROWTH (REVENUE vs PROFIT TOGGLE) ---
            st.divider()

            # 1. Toggle Selection
            # We use a horizontal radio button to switch between metrics
            toggle_col1, toggle_col2 = st.columns([1, 2])
            with toggle_col1:
                metric_choice = st.radio("Select View:", ["Revenue", "Profit"], horizontal=True)

            # Map the choice to the correct yfinance row name
            data_row = 'Total Revenue' if metric_choice == "Revenue" else 'Net Income'
            chart_color = '#3a7bd5' if metric_choice == "Revenue" else '#00cc96' # Blue for Rev, Green for Profit

            st.markdown(f"### 💰 {metric_choice} Analysis: {selected_stock}")

            rev_col1, rev_col2 = st.columns(2)

            with rev_col1:
                # --- Annual Chart ---
                annual_fin = ticker_obj.financials
                if not annual_fin.empty and data_row in annual_fin.index:
                    # Fetch and clean data
                    data_a = annual_fin.loc[data_row].dropna().head(5)
                    df_a = data_a.reset_index()
                    df_a.columns = ['Year', 'Value']
                    
                    # Format Year and convert to Crores (10^7)
                    df_a['Year'] = df_a['Year'].dt.strftime('%Y')
                    df_a['Value_Cr'] = df_a['Value'] / 10**7
                    df_a = df_a.sort_values('Year')

                    fig_a = px.bar(df_a, x='Year', y='Value_Cr', 
                                title=f"Annual {metric_choice} (₹ Cr)",
                                text_auto='.2s')
                    
                    fig_a.update_traces(marker_color=chart_color)
                    fig_a.update_xaxes(type='category', title_text="Fiscal Year")
                    st.plotly_chart(fig_a, use_container_width=True)
                else:
                    st.info(f"Annual {metric_choice} data unavailable.")

            with rev_col2:
                # --- Quarterly Chart ---
                qtr_fin = ticker_obj.quarterly_financials
                if not qtr_fin.empty and data_row in qtr_fin.index:
                    # Fetch and clean data
                    data_q = qtr_fin.loc[data_row].dropna().head(4)
                    df_q = data_q.reset_index()
                    df_q.columns = ['Quarter', 'Value']
                    
                    # Format Quarter and convert to Crores
                    df_q['Quarter'] = df_q['Quarter'].dt.strftime('%b %Y')
                    df_q['Value_Cr'] = df_q['Value'] / 10**7
                    df_q = df_q.sort_index(ascending=False) # Oldest to newest

                    fig_q = px.bar(df_q, x='Quarter', y='Value_Cr', 
                                title=f"Quarterly {metric_choice} (₹ Cr)",
                                text_auto='.2s')
                    
                    fig_q.update_traces(marker_color=chart_color)
                    fig_q.update_xaxes(type='category', title_text="Quarter Ending")
                    st.plotly_chart(fig_q, use_container_width=True)
                else:
                    st.info(f"Quarterly {metric_choice} data unavailable.")

        else:
            st.error("Request token not found in URL.")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please complete the sidebar configuration and login to view your dashboard.")