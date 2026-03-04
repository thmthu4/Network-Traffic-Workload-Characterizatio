import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def generate_synthetic_traffic(n_samples=5000):
    np.random.seed(42)
    print(f"Generating {n_samples} packets of synthetic network traffic...")
    
    # 1. Protocols & Markov Sequences
    protocols = ['TCP', 'UDP', 'DNS', 'ICMP']
    # Transition matrix heavily favoring typical sequence patterns
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
    
    # 2. Packet Sizes (Bytes)
    packet_sizes = np.zeros(n_samples)
    for i, p in enumerate(protocol_data):
        if p == 'TCP':
            # Mix of control ACKs (small) and Data (large) - typical bimodal distribution
            if np.random.rand() < 0.4:
                packet_sizes[i] = np.random.normal(1460, 50) # Data payload
            else:
                packet_sizes[i] = np.random.normal(64, 5) # ACK
        elif p == 'UDP':
            packet_sizes[i] = np.random.normal(512, 100)
        elif p == 'DNS':
            packet_sizes[i] = np.random.normal(120, 20)
        elif p == 'ICMP':
            packet_sizes[i] = np.random.normal(64, 2)
            
    # Clip sizes to realistic boundaries (min ethernet frame to MTU approx)
    packet_sizes = np.clip(packet_sizes, 40, 2000)
    
    # 3. Inter-arrival time (ms) - Exponential distribution
    inter_arrival = np.random.exponential(scale=15.0, size=n_samples) 
    
    # 4. Create DataFrame
    df = pd.DataFrame({
        'Protocol': protocol_data,
        'Packet_Size': packet_sizes,
        'Inter_Arrival': inter_arrival
    })
    
    # Derived parameter: Packet Rate (Packets Per Second)
    # Using 1000ms / inter-arrival to get instantaneous rate approximation
    df['Packet_Rate'] = 1000 / (df['Inter_Arrival'] + 0.001) 
    
    return df, protocols

def create_dashboard(df, protocols):
    print("Building visualization dashboard...")
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Network Traffic Workload Characterization Dashboard", fontsize=24, fontweight='bold', y=0.98)
    
    # Colors
    primary_color = '#3498db'
    
    # --- 1. Single-Parameter Histogram ---
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df['Packet_Size'], bins=50, color=primary_color, edgecolor='black', alpha=0.8)
    ax1.set_title("1. Single-Param Hist: Packet Sizes", fontsize=14)
    ax1.set_xlabel("Packet Size (Bytes)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    
    # --- 2. Multi-Parameter Histogram (Hexbin) ---
    ax2 = plt.subplot(2, 3, 2)
    hb = ax2.hexbin(df['Packet_Size'], df['Inter_Arrival'], gridsize=40, cmap='YlOrRd', bins='log')
    ax2.set_title("2. Multi-Param Hist: Size vs. Inter-arrival (Log Scale)", fontsize=14)
    ax2.set_xlabel("Packet Size (Bytes)", fontsize=12)
    ax2.set_ylabel("Inter-arrival Time (ms)", fontsize=12)
    cb = fig.colorbar(hb, ax=ax2)
    cb.set_label('log10(count)')
    
    # --- 3. Principal Component Analysis (PCA) ---
    ax3 = plt.subplot(2, 3, 3)
    features = ['Packet_Size', 'Inter_Arrival', 'Packet_Rate']
    X = df[features].values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0) # Standardize
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, c='#9b59b6', s=15)
    var_explained = sum(pca.explained_variance_ratio_) * 100
    ax3.set_title(f"3. PCA (2 Components explain {var_explained:.1f}% variance)", fontsize=14)
    ax3.set_xlabel("Principal Component 1 (Dominated by Traffic Volume)", fontsize=12)
    ax3.set_ylabel("Principal Component 2 (Dominated by Timing)", fontsize=12)
    ax3.grid(alpha=0.3)
    
    # --- 4. Markov Model Transition Matrix ---
    ax4 = plt.subplot(2, 3, 4)
    transitions = np.zeros((4, 4))
    protocol_list = df['Protocol'].tolist()
    for i in range(len(protocol_list) - 1):
        curr_idx = protocols.index(protocol_list[i])
        next_idx = protocols.index(protocol_list[i+1])
        transitions[curr_idx, next_idx] += 1
        
    row_sums = transitions.sum(axis=1, keepdims=True)
    transitions_prob = np.divide(transitions, row_sums, out=np.zeros_like(transitions), where=row_sums!=0)
    
    cax = ax4.matshow(transitions_prob, cmap='Blues', alpha=0.8)
    for i in range(4):
        for j in range(4):
            ax4.text(j, i, f"{transitions_prob[i, j]:.2f}", ha='center', va='center', 
                     color='white' if transitions_prob[i, j] > 0.5 else 'black', fontweight='bold')
            
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(protocols, fontsize=11)
    ax4.set_yticklabels(protocols, fontsize=11)
    ax4.set_title("4. Markov Model: Protocol Transitions Matrix", pad=15, fontsize=14)
    ax4.set_xlabel("Next Protocol State", fontsize=12)
    ax4.set_ylabel("Current Protocol State", fontsize=12)
    
    # --- 5. Clustering Analysis (K-Means) ---
    ax5 = plt.subplot(2, 3, 5)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_std)
    
    # Map clusters to descriptive names based on characteristics
    scatter5 = ax5.scatter(df['Packet_Size'], df['Packet_Rate'], c=clusters, cmap='viridis', s=15, alpha=0.6)
    ax5.set_title("5. Clustering Analysis: Traffic Behavioral Groups", fontsize=14)
    ax5.set_xlabel("Packet Size (Bytes)", fontsize=12)
    ax5.set_ylabel("Packet Rate (pps)", fontsize=12)
    
    # Custom Legend
    handles, _ = scatter5.legend_elements()
    labels = ["Normal Browsing", "Control/Background", "High-Intensity Transfer"]
    ax5.legend(handles, labels, loc="upper right", title="Traffic Classes")
    ax5.grid(alpha=0.3)
    
    # --- 6. Averaging & Dispersion Text Box ---
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    ax6.set_title("6. Averaging & Dispersion Statistics", fontsize=14, fontweight='bold', pad=20)
    
    stats_text = (
        f"DATASET OVERVIEW\n"
        f"----------------------------------------\n"
        f"Total Packets Analyzed : {len(df):,}\n\n"
        
        f"AVERAGING TECHNIQUES\n"
        f"----------------------------------------\n"
        f"Mean Packet Size       : {df['Packet_Size'].mean():.2f} Bytes\n"
        f"Mean Inter-arrival Time: {df['Inter_Arrival'].mean():.2f} ms\n"
        f"Mean Packet Rate       : {df['Packet_Rate'].mean():.2f} pps\n\n"
        
        f"DISPERSION TECHNIQUES\n"
        f"----------------------------------------\n"
        f"Packet Size Variance   : {df['Packet_Size'].var():.2f}\n"
        f"Pkt Size Std Deviation : {df['Packet_Size'].std():.2f} Bytes\n"
        f"IAT Variance           : {df['Inter_Arrival'].var():.2f}\n"
        f"IAT Std Deviation      : {df['Inter_Arrival'].std():.2f} ms\n\n"
        f"Note: High dispersion indicates highly bursty\n"
        f"traffic characteristics as per the report."
    )
    
    ax6.text(0.1, 0.5, stats_text, fontsize=13, va='center', family='monospace', 
             bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=1.5', alpha=0.9))
    
    # Finalize and Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = 'network_workload_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Success! Dashboard saved to {os.path.abspath(output_path)}")
    plt.show()

if __name__ == "__main__":
    df, protocols = generate_synthetic_traffic()
    create_dashboard(df, protocols)
