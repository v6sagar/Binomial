import streamlit as st
import numpy as np
import graphviz as gv

# Binomial Call Option Pricing Model Function
def binomial_call_full(S_ini, K, T, r, u, d, N):
    dt = T / N  # Time step
    p = (np.exp(r * dt) - d) / (u - d)  # Risk neutral probability
    C = np.zeros([N + 1, N + 1])  # Call option prices
    S = np.zeros([N + 1, N + 1])  # Underlying stock prices

    # Populate the last row of the binomial tree (option values at maturity)
    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** i) * (d ** (N - i))
        C[N, i] = max(S[N, i] - K, 0)

    # Backward induction for option pricing
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            S[j, i] = S_ini * (u ** i) * (d ** (j - i))
            C[j, i] = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])

    return C[0, 0], C, S

# Function to visualize the binomial tree using graphviz
def visualize_binomial_tree(tree, label):
    dot = gv.Digraph()

    # Iterate through each level of the tree
    for level in range(tree.shape[0]):
        for i in range(level + 1):
            node_id = f"{label}{level}_{i}"
            dot.node(node_id, f"{tree[level, i]:.2f}")

            # Connect to the next level
            if level < tree.shape[0] - 1:
                dot.edge(node_id, f"{label}{level+1}_{i}")      # Downward branch
                dot.edge(node_id, f"{label}{level+1}_{i+1}")    # Upward branch

    return dot

# Streamlit app
st.title("Binomial Option Pricing Model")

# User inputs
S_ini = st.number_input("Initial Stock Price (S₀)", min_value=0.0, value=100.0)
K = st.number_input("Strike Price (K)", min_value=0.0, value=100.0)
T = st.number_input("Time to Maturity (T) in years", min_value=0.0, value=1.0)
r = st.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05)
u = st.number_input("Upward Movement Factor (u)", min_value=0.0, value=1.1)
d = st.number_input("Downward Movement Factor (d)", min_value=0.0, value=0.9)
N = st.number_input("Number of Steps (N)", min_value=1, value=3)

# Calculate the option price and binomial trees
if st.button("Calculate"):
    call_price, C_tree, S_tree = binomial_call_full(S_ini, K, T, r, u, d, N)

    st.write(f"Option Price: {call_price:.2f}")

    # Visualize the underlying stock price tree
    st.subheader("Underlying Stock Price Tree")
    stock_tree_graph = visualize_binomial_tree(S_tree, "S")
    st.graphviz_chart(stock_tree_graph)
    
    # Visualize the call option price tree
    st.subheader("Call Option Price Tree")
    call_tree_graph = visualize_binomial_tree(C_tree, "C")
    st.graphviz_chart(call_tree_graph)
