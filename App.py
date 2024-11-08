import streamlit as st
import numpy as np
import pandas as pd
from BlackScholes import BlackScholes
from BinomialTree import BinomialTree
from MonteCarlo import MonteCarloSimulation
import plotly.express as px
import plotly.graph_objects as go
import io

# Page configuration
st.set_page_config(
    page_title="Option Pricing App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
.metric-container {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
}
.metric-value {
    font-size: 2em;
    color: #4a4a4a;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("##### Made by Jabir Hussain")
    st.title("Option Parameters")
    S = st.slider("Current Asset Price (S)", min_value=0.01, max_value=500.0, value=100.0, help="The current price of the underlying asset.")
    K = st.slider("Strike Price (K)", min_value=0.01, max_value=500.0, value=100.0, help="The price at which the option can be exercised.")
    T = st.slider("Time to Maturity (T in years)", min_value=0.01, max_value=5.0, value=1.0, help="Time remaining until the option expires.")
    sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.2, help="Annualized volatility of the asset's returns.")
    r = st.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.2, value=0.05, help="Risk-free rate of return over the option's life.")
    option_type = st.selectbox("Option Type", options=["Call", "Put"], index=0)
    dark_mode = st.checkbox("Enable Dark Mode")

    # Adjust theme based on dark mode
    if dark_mode:
        st.markdown("""
        <style>
        body {
            background-color: #2e2e2e;
            color: #f0f0f0;
        }
        .metric-container {
            background-color: #444444;
            color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .metric-container {
            background-color: #f0f0f0;
            color: #000000;
        }
        </style>
        """, unsafe_allow_html=True)

        # Watermark CSS
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 12px;
        color: rgba(150, 150, 150, 0.6);  /* Light gray with some transparency */
        z-index: 1000;
    }
    </style>
    <div class="watermark">
        Created by Your Name
    </div>
""", unsafe_allow_html=True)


# Main content
st.title("Option Pricing App")

# Create tabs for navigation
tab1, tab2, tab3, tab4 = st.tabs(["Pricing", "Greeks", "Simulation", "Scenarios"])

with tab1:
    st.header("Option Pricing")
    # Black-Scholes pricing
    bs = BlackScholes(S, K, T, r, sigma)
    call_price = bs.call_price()
    put_price = bs.put_price()

    # Display prices
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-container"><div class="metric-value">${call_price:.2f}</div><div>European Call Option Price</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><div class="metric-value">${put_price:.2f}</div><div>European Put Option Price</div></div>', unsafe_allow_html=True)

    # Binomial Tree pricing for American options
    bt = BinomialTree(S, K, T, r, sigma, option_type=option_type.lower())
    american_price = bt.price()

    st.markdown(f'<div class="metric-container"><div class="metric-value">${american_price:.2f}</div><div>American {option_type} Option Price (Binomial Tree)</div></div>', unsafe_allow_html=True)

    # Breakdown of Black-Scholes calculations
    st.subheader("Calculation Breakdown")
    bs._calculate_d1_d2()
    st.markdown(f"**d₁** = {bs.d1:.4f}")
    st.markdown(f"**d₂** = {bs.d2:.4f}")

with tab2:
    st.header("Option Greeks")
    # Calculate Greeks
    delta = bs.delta(option_type.lower())
    gamma = bs.gamma()
    vega = bs.vega()
    theta = bs.theta(option_type.lower())
    rho = bs.rho(option_type.lower())

    # Display Greeks
    greeks = {
        'Greek': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        'Value': [delta, gamma, vega, theta, rho]
    }
    greeks_df = pd.DataFrame(greeks)
    st.table(greeks_df)

    # Interactive heatmap of option prices
    st.subheader("Option Price Heatmap")
    spot_prices = np.linspace(S * 0.5, S * 1.5, 50)
    volatilities = np.linspace(sigma * 0.5, sigma * 1.5, 50)
    Spot, Vol = np.meshgrid(spot_prices, volatilities)

    def compute_option_prices(Spot, Vol):
        prices = np.zeros(Spot.shape)
        for i in range(Spot.shape[0]):
            for j in range(Spot.shape[1]):
                bs_temp = BlackScholes(Spot[i, j], K, T, r, Vol[i, j])
                if option_type.lower() == 'call':
                    prices[i, j] = bs_temp.call_price()
                else:
                    prices[i, j] = bs_temp.put_price()
        return prices

    prices = compute_option_prices(Spot, Vol)
    fig = px.imshow(
        prices,
        x=spot_prices,
        y=volatilities,
        labels={'x': 'Spot Price', 'y': 'Volatility', 'color': f'{option_type} Price'},
        aspect='auto'
    )
    fig.update_layout(title=f"{option_type} Option Price Heatmap", coloraxis_colorbar=dict(title="Price"))
    st.plotly_chart(fig)

with tab3:
    st.header("Monte Carlo Simulation")
    if st.button("Run Simulation"):
        mc_sim = MonteCarloSimulation(S, T, r, sigma)
        price_paths = mc_sim.simulate()

        # Plot simulated paths
        fig = go.Figure()
        time_steps = np.linspace(0, T, mc_sim.steps + 1)
        for i in range(100):  # Plot first 100 paths
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=price_paths[:, i],
                mode='lines',
                line=dict(width=1),
                showlegend=False
            ))
        fig.update_layout(title="Simulated Stock Price Paths", xaxis_title="Time (Years)", yaxis_title="Stock Price")
        st.plotly_chart(fig)

with tab4:
    st.header("Scenario Analysis")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Scenario 1")
        S1 = st.number_input("Asset Price (S1)", value=S, key='S1')
        K1 = st.number_input("Strike Price (K1)", value=K, key='K1')
        T1 = st.number_input("Time to Maturity (T1)", value=T, key='T1')
        sigma1 = st.number_input("Volatility (σ1)", value=sigma, key='sigma1')
        r1 = st.number_input("Interest Rate (r1)", value=r, key='r1')
        option_type1 = st.selectbox("Option Type (1)", options=["Call", "Put"], key='option_type1')

        bs1 = BlackScholes(S1, K1, T1, r1, sigma1)
        price1 = bs1.call_price() if option_type1.lower() == 'call' else bs1.put_price()
        st.write(f"{option_type1} Option Price: ${price1:.2f}")

    with col6:
        st.subheader("Scenario 2")
        S2 = st.number_input("Asset Price (S2)", value=S, key='S2')
        K2 = st.number_input("Strike Price (K2)", value=K, key='K2')
        T2 = st.number_input("Time to Maturity (T2)", value=T, key='T2')
        sigma2 = st.number_input("Volatility (σ2)", value=sigma, key='sigma2')
        r2 = st.number_input("Interest Rate (r2)", value=r, key='r2')
        option_type2 = st.selectbox("Option Type (2)", options=["Call", "Put"], key='option_type2')

        bs2 = BlackScholes(S2, K2, T2, r2, sigma2)
        price2 = bs2.call_price() if option_type2.lower() == 'call' else bs2.put_price()
        st.write(f"{option_type2} Option Price: ${price2:.2f}")

# Download report
st.subheader("Download Option Report")
report_data = {
    'Parameter': ['Asset Price', 'Strike Price', 'Time to Maturity', 'Volatility', 'Interest Rate'],
    'Value': [S, K, T, sigma, r]
}
report_df = pd.DataFrame(report_data)
greeks_df['Parameter'] = greeks_df['Greek']
report_df = pd.concat([report_df, greeks_df[['Parameter', 'Value']]], ignore_index=True)

csv = report_df.to_csv(index=False)
st.download_button(
    label="Download Report as CSV",
    data=csv,
    file_name='option_pricing_report.csv',
    mime='text/csv'
)
