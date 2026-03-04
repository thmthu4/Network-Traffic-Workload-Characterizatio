import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Network Workload Characterization", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner, modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .explanation-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_traffic(n_samples=5000):
    np.random.seed(42)
    protocols = ['TCP', 'UDP', 'DNS', 'ICMP']
    transition_matrix = {
        'TCP':  [0.70, 0.10, 0.15, 0.05],
        'UDP':  [0.20, 0.60, 0.10, 0.10],
        'DNS':  [0.80, 0.10, 0.05, 0.05],
        'ICMP': [0.40, 0.20, 0.10, 0.30]
    }
    
    seq = ['TCP']
    for _ in range(1, n_samples):
        prev = seq[-1]
        next_prot = np.random.choice(protocols, p=transition_matrix[prev])
        seq.append(next_prot)
    protocol_data = np.array(seq)
    
    packet_sizes = np.zeros(n_samples)
    for i, p in enumerate(protocol_data):
        if p == 'TCP':
            if np.random.rand() < 0.4:
                packet_sizes[i] = np.random.normal(1460, 50) 
            else:
                packet_sizes[i] = np.random.normal(64, 5) 
        elif p == 'UDP':
            packet_sizes[i] = np.random.normal(512, 100)
        elif p == 'DNS':
            packet_sizes[i] = np.random.normal(120, 20)
        elif p == 'ICMP':
            packet_sizes[i] = np.random.normal(64, 2)
            
    packet_sizes = np.clip(packet_sizes, 40, 2000)
    inter_arrival = np.random.exponential(scale=15.0, size=n_samples) 
    
    df = pd.DataFrame({
        'Protocol': protocol_data,
        'Packet_Size': packet_sizes,
        'Inter_Arrival': inter_arrival
    })
    
    df['Packet_Rate'] = 1000 / (df['Inter_Arrival'] + 0.001) 
    return df, protocols

# --- HEADER ---
st.markdown('<div class="main-header" style="text-align: center;">Network Traffic Workload Characterization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header" style="text-align: center;">Analytical & Statistical Techniques for Performance Analysis</div>', unsafe_allow_html=True)

# --- SIDEBAR & CONFIGURATION ---
st.sidebar.header("Configuration")
st.sidebar.markdown("Adjust the number of packets simulated for the dashboard.")
n_samples = st.sidebar.slider("Number of Packets", 1000, 20000, 5000, 1000)

df, protocols = generate_synthetic_traffic(n_samples)

# --- TABS FOR PRESENTATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Averaging & Dispersion", 
    "Histograms", 
    "Dimensionality Reduction (PCA)", 
    "Protocol State Transitions", 
    "Traffic Clustering"
])

# Utility function for chart formatting
def format_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=15, fontweight='bold', color='#1e293b')
    ax.set_xlabel(xlabel, fontweight='medium', color='#475569')
    ax.set_ylabel(ylabel, fontweight='medium', color='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, linestyle='--')

with tab1:
    st.markdown("""
    <div class="explanation-box">
        <strong>Averaging & Dispersion</strong> provides the baseline characteristics of network workload. 
        <ul>
            <li><strong>Averaging (Means)</strong>: Identifies the <em>typical</em> behavior of the traffic (intensity and volume).</li>
            <li><strong>Dispersion (Variance/Std Dev)</strong>: Measures how much the workload varies over time. High dispersion indicates highly variable, bursty network traffic rather than a smooth, constant flow.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Averaging Techniques (Central Tendencies)**")
        c1, c2 = st.columns(2)
        c1.metric("Mean Packet Size", f"{df['Packet_Size'].mean():.1f} Bytes")
        c2.metric("Mean Inter-arrival", f"{df['Inter_Arrival'].mean():.1f} ms")
        
        c3, c4 = st.columns(2)
        c3.metric("Mean Packet Rate", f"{df['Packet_Rate'].mean():.1f} pkt/sec")
        c4.metric("Dataset Size", f"{len(df):,} Packets")

    with col2:
        st.markdown("**Dispersion Techniques (Traffic Variance)**")
        c1, c2 = st.columns(2)
        c1.metric("Packet Size Variance", f"{df['Packet_Size'].var():.0f}")
        c2.metric("Size Std Deviation", f"{df['Packet_Size'].std():.1f} Bytes")
        
        c3, c4 = st.columns(2)
        c3.metric("Inter-arrival Variance", f"{df['Inter_Arrival'].var():.1f}")
        c4.metric("IAT Std Deviation", f"{df['Inter_Arrival'].std():.1f} ms")


with tab2:
    st.markdown("""
    <div class="explanation-box">
        <strong>Histogram Analysis</strong> visuals help us move beyond simple averages to understand the actual distribution of network workload parameters.
        <ul>
            <li><strong>Single-Parameter Histogram</strong>: Reveals frequencies of individual parameters (e.g., distinguishing between small control packets vs. large data payloads).</li>
            <li><strong>Multi-Parameter Histogram</strong>: Uncovers correlations between two parameters simultaneously (e.g., do large packets arrive closer together or further apart?).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    with c1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.hist(df['Packet_Size'], bins=40, color='#3b82f6', edgecolor='white')
        format_ax(ax1, "Distribution of Packet Sizes", "Packet Size (Bytes)", "Frequency")
        st.pyplot(fig1)
        
    with c2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        hb = ax2.hexbin(df['Packet_Size'], df['Inter_Arrival'], gridsize=30, cmap='YlOrRd', bins='log')
        format_ax(ax2, "Multi-Parameter: Size vs. Inter-arrival", "Packet Size (Bytes)", "Inter-arrival Time (ms)")
        plt.colorbar(hb, ax=ax2, label='log10(count)', fraction=0.046, pad=0.04)
        st.pyplot(fig2)

with tab3:
    st.markdown("""
    <div class="explanation-box">
        <strong>Principal Component Analysis (PCA)</strong> is an advanced mathematical technique used to simplify complex workload data.
        As the number of parameters increases, analysis becomes difficult. PCA reduces the "dimensionality" of the data 
        by generating new variables (Principal Components) that still capture the majority of the variance (behavior) of the original traffic.
    </div>
    """, unsafe_allow_html=True)
    
    features = ['Packet_Size', 'Inter_Arrival', 'Packet_Rate']
    X = df[features].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    var_explained = sum(pca.explained_variance_ratio_) * 100
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Key Insight**")
        st.write(f"By reducing our parameters to just 2 principal components, we have retained **{var_explained:.1f}%** of the total variance in the network behavior.")
        st.write("This proves that network traffic is dominated by just a few key underlying characteristics (like timing and volume).")
        
    with col2:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c='#8b5cf6', s=20, edgecolor='white', linewidth=0.5)
        format_ax(ax3, "Traffic Projected onto Principal Components", "Principal Component 1", "Principal Component 2")
        st.pyplot(fig3)

with tab4:
    st.markdown("""
    <div class="explanation-box">
        <strong>Markov Models</strong> are used to model the sequential behavior of network protocols. 
        Rather than treating each packet as isolated, a Markov model treats each protocol as a "state". 
        The matrix below shows the probability of transitioning from one protocol state to the next, revealing normal sequential behavior (like DNS queries frequently preceding TCP handshakes).
    </div>
    """, unsafe_allow_html=True)
    
    transitions = np.zeros((4, 4))
    protocol_list = df['Protocol'].tolist()
    for i in range(len(protocol_list) - 1):
        curr_idx = protocols.index(protocol_list[i])
        next_idx = protocols.index(protocol_list[i+1])
        transitions[curr_idx, next_idx] += 1
        
    row_sums = transitions.sum(axis=1, keepdims=True)
    transitions_prob = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        cax = ax4.matshow(transitions_prob, cmap='Blues', alpha=0.8)
        for i in range(4):
            for j in range(4):
                ax4.text(j, i, f"{transitions_prob[i, j]:.2f}", ha='center', va='center', 
                         color='white' if transitions_prob[i, j] > 0.5 else 'black', fontweight='bold')
        
        ax4.set_xticks(range(4))
        ax4.set_yticks(range(4))
        ax4.set_xticklabels(protocols)
        ax4.set_yticklabels(protocols)
        ax4.set_title("Protocol Transition Matrix", pad=20, fontweight='bold')
        ax4.xaxis.set_ticks_position('bottom')
        st.pyplot(fig4)
        
    with col2:
        st.markdown("**How to Read This**")
        st.write("Read from the **Row (Current State)** to the **Column (Next State)**.")
        st.write("Example: The number in the 'DNS' row and 'TCP' column represents the probability that a DNS query is immediately followed by a TCP packet.")

with tab5:
    st.markdown("""
    <div class="explanation-box">
        <strong>Clustering Analysis</strong> groups network traffic samples based on mathematical similarities in their features (like size and rate). 
        This is an incredibly powerful technique for identifying distinct traffic profiles and detecting anomalies, without needing to manually define thresholds.
    </div>
    """, unsafe_allow_html=True)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_std)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        scatter = ax5.scatter(df['Packet_Size'], df['Packet_Rate'], c=clusters, cmap='tab10', s=25, alpha=0.7, edgecolor='white', linewidth=0.5)
        format_ax(ax5, "K-Means Clustering: Traffic Profiles", "Packet Size (Bytes)", "Packet Rate (pps)")
        
        # Legend
        handles, _ = scatter.legend_elements()
        labels = ["High-Intensity Transfer", "Normal Active Data", "Intermittent Background"]
        ax5.legend(handles, labels, loc="upper right", title="Identified Clusters", framealpha=0.9)
        st.pyplot(fig5)
        
    with col2:
        st.markdown("**Identified Profiles**")
        st.write("The algorithm automatically identified grouping without being told what they are:")
        st.write("1. **High-Intensity Transfer**: Fast packet rates, potentially large packets.")
        st.write("2. **Normal Active Data**: Average sizes, average rates.")
        st.write("3. **Intermittent Background**: Small control packets, low rates.")
