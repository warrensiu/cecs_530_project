# LLaMA Token-Generation Latency Benchmarking

## 📌 Project Overview
This repository contains the benchmarking harness and performance analysis for investigating **time-per-generated-token** in LLaMA-style models. While aggregate throughput is a common metric, user-perceived quality in interactive LLM applications is dominated by per-token latency. The project focuses on decomposing inference latency into its architectural components to identify hardware and system-level bottlenecks.

## 🛠️ Key Objectives
* **Instrument the Inference Stack:** Measure latency across various stages of the LLaMA pipeline.
* **Decompose Latency:** Attribute time spent to specific components such as Attention, MLP, and KV-cache.
* **Scaling Analysis:** Study how latency scales with sequence length and model size.
* **Architectural Reasoning:** Connect measured behavior to hardware limits like memory bandwidth saturation and arithmetic intensity.
* **Optimization Proposals:** Propose and estimate the impact of concrete system-level improvements.

---

## 🚀 Experimental Scope
As per project constraints, this benchmark focuses on **autoregressive token generation** under the following conditions:

* **Batch Size:** 1 (required baseline).
* **Input/Output Length:** Tested with fixed prompt and generation lengths (e.g., 128 tokens).
* **Architecture:** LLaMA-style Transformer models.
* **Hardware:** Benchmarked on AMD Ryzen 7 5700X 8-Core Processor, NVIDIA GeForce RTX 4070.

---

## 📊 Benchmarking Methodology

### 1. Harness Design
The benchmarking harness is designed for repeatability and includes:
* **Warm-up Runs:** To stabilize hardware states and caches before recording.
* **Multiple Trials:** Ensuring statistical significance with outlier handling.
* **Metrics Captured:** * **Time to First Token (TTFT):** Prompt processing latency.
    * **Per-Token Latency:** Steady-state generation time.
    * **End-to-End Response Time:** Total round-trip latency.

### 2. Latency Decomposition
We instrumented the following pipeline stages to identify the critical path:
* Embedding lookup
* Attention (QKV, softmax, projection)
* KV-cache read/write operations
* MLP (Multi-Layer Perceptron)
* LayerNorm and Residual connections
* Sampling and decoding logic
* Framework overhead (kernel launch, synchronization)

---

## 🔍 Analysis & Bottlenecks

### Scaling Analysis
This repository includes scaling plots analyzing:
* **Sequence Length:** Impact of KV-cache growth on attention bandwidth.
* **Model Size:** Comparisons across different parameter counts to identify bottleneck inflection points.

### Architectural Implications
Our analysis maps measurements to architectural causes, specifically:
* **Memory Bandwidth Saturation:** Identifying when the model becomes memory-bound during decoding.
* **Arithmetic Intensity:** Analyzing why FLOPs utilization is often low during autoregressive generation.
* **Pipeline Bubbles & Sync Overhead:** Quantifying framework-level inefficiencies.

---

## 📁 Project Structure
* `benchmark.ipynb`: Primary notebook containing model instrumentation and data collection.
* `/images`: Raw images of results from benchmark.ipynb
## 🔧 Installation & Usage

### Prerequisites
* Python 3.10+
* PyTorch / Transformers
* Profiling tools (prefered Jupyter Notebook)
* `pip install torch numpy matplotlib nvidia-ml-py`  

### Running the Benchmark
1. Open Jupyter Notebook 
2. Import benchmark.ipynb
3. Open `benchmark.ipynb`
4. Execute the cells to initialize the model
5. Start the measurement harness

---

## 👤 Author
Stephen Shao, Huayue Ding, Warren Siu





