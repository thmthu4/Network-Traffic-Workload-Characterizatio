# Network Traffic Workload Characterization

This project applies the concepts from Chapter 6 of Raj Jain's book, *The Art of Computer Systems Performance Analysis*, to characterize network traffic workloads. 

By analyzing network packet parameters (such as packet size, inter-arrival time, packet rate, and protocol), this application demonstrates advanced mathematical and statistical techniques for understanding network behavior without relying on simple averages alone.

## 🚀 Features & Techniques Used

This project dynamically synthesizes realistic network traffic data (simulating TCP, UDP, DNS, and ICMP protocols) and visualizes it using the following analytical techniques:

1. **Averaging & Dispersion**: Calculates central tendencies (Mean) and traffic burstiness (Variance and Standard Deviation).
2. **Histograms**: Visualizes the distribution of single parameters (Packet Size) and the relationship between multiple parameters (Size vs. Inter-arrival time).
3. **Principal Component Analysis (PCA)**: A dimensionality reduction technique that compresses complex multi-parameter data into just 2 components while preserving the majority of original variance.
4. **Markov Models**: Analyzes sequential behavior by calculating the transition probabilities between different networking protocols.
5. **K-Means Clustering**: Uses machine learning to group traffic into behavioral profiles (e.g., High-Intensity Transfer, Normal Active Data) for anomaly detection.

## 📁 Project Structure

- `web_dashboard.py`: An interactive, professional web application built with Streamlit that explains and visualizes the workload characterization techniques through an intuitive UI.
- `visualize_traffic.py`: A standalone Python script that generates the data and saves all charts onto a single high-resolution static image (`network_workload_dashboard.png`).
- `read_docx.py` (optional Utility): Script to extract text content directly from the original `.docx` proposal files.

## 🛠️ Installation & Setup

Ensure you have Python 3 installed. Then, install the required numerical and visualization libraries:

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

## 🎮 How to Run

### Option 1: The Interactive Web Dashboard (Recommended)
This launches a beautiful, tabbed web application with educational explanations for each technique and a slider to dynamically adjust the number of simulated packets.

```bash
streamlit run web_dashboard.py
```
*The dashboard will automatically open in your default web browser at `http://localhost:8501`.*

### Option 2: The Static PNG Script
If you just want to generate the high-resolution dashboard image without starting a web server:

```bash
python visualize_traffic.py
```
*This will generate `network_workload_dashboard.png` in your current directory.*
