# 🧩 **1. What is KV Cache?**

Transformers store:
- **K = Key vectors**
- **V = Value vectors**

for each token during decoding.

This avoids recomputing attention on previous tokens.

---

# ✅ **2. Why KV Cache is Essential**

Without KV cache:  
→ Each new token requires O(N²) attention.

With KV cache:  
→ Each new token requires O(N) compute.

---

```mermaid
flowchart TD
    A["Token 1 → Compute K1,V1"] --> B["Store in KV Cache"]
    C["Token 2 → Compute K2,V2"] --> B
    D["Token 3"] --> E["Read existing KV (1..2)"]
    E --> F["Compute Attention"]
```

---

# ✅ **3. KV Cache Memory Problem**

KV cache grows linearly with sequence length × layers × batch size.

Long prompts (100k–1M tokens) → KV cache dominates GPU memory.

---

# ✅ **4. Evolution of KV Cache Management**

```mermaid
flowchart TD
    A["Naive KV Cache"] --> B["Static Preallocated KV"]
    B --> C["Paged KV Cache"]
    C --> D["PagedAttention Kernel"]
    D --> E["LMCache (Multi-level KV)"]
    E --> F["Parameter & Activation Disaggregation"]
    E --> G["PD: Prefill/Decode Disaggregation"]
```

---

# ✅ **5. Static KV Cache (Early Implementations)**

- Big contiguous buffer
- Fragmentation
- Hard to support multi-user batches
- OOM common

---

# ✅ **6. Paged KV Cache (vLLM Breakthrough)**

Break KV cache into **fixed-size blocks**, just like OS memory pages.

```mermaid
flowchart TD
    A["Logical KV Cache"] --> B["Page Table"]
    B --> C["Block 0"]
    B --> D["Block 1"]
    B --> E["Block 2"]
    B --> F["Block 3"]
```

✅ Avoids fragmentation  
✅ Allows reuse of freed KV pages  
✅ Enables continuous batching  

---

# ✅ **7. PagedAttention — Page-aware GPU Kernel**

PagedAttention uses:
- Block pointers
- Coalesced block loads
- Continuous block reuse

```mermaid
flowchart LR
    A["Token Embedding"] --> B["KV Generation"]
    B --> C["Paged KV Blocks"]
    C --> D["Attention Kernel (PagedAccess)"]
    D --> E["Next Token"]
```

✅ High throughput  
✅ Enables 4–10× more concurrent users  

---

# ✅ **8. LMCache — Multi‑Level KV Cache (GPU → CPU → NVMe)**

When prompts exceed GPU memory, we need a **tiered KV storage hierarchy**:

```mermaid
flowchart TD
    A["GPU KV Cache - L1"] --> B["CPU KV Cache - L2"]
    B --> C["NVMe KV Cache - L3"]
```

L1 = Fastest, smallest  
L2 = Medium capacity  
L3 = Very large but slow  

LMCache implements:
- Prefetching  
- Eviction  
- Async background transfers  

✅ Enables million‑token context inference  

---

# ✅ **9. Parameter & Activation Disaggregation**

Goal: Store as little as possible on GPU.

### ✅ Parameter Disaggregation
Weights live in:
- Host RAM
- NVMe SSD
- Remote parameter servers

### ✅ Activation / KV Disaggregation
KV lives across:
- GPU
- CPU
- NVMe
- Remote nodes

```mermaid
flowchart TD
    A["GPU Compute"] --> B["Is Layer Weight Local?"]
    B -->|No| C["Fetch Weight from CPU/NVMe"]
    B -->|Yes| D["Use Local Weight"]

    D --> E["Need KV Block?"]
    E -->|GPU| F["Use GPU KV"]
    E -->|CPU| G["Fetch from CPU"]
    E -->|NVMe| H["Fetch from NVMe"]
    E -->|Remote| I["Fetch from Remote Node"]

    F --> J["Attention Compute"]
    G --> J
    H --> J
    I --> J
```

✅ Enables **trillion‑parameter scale**  
✅ Works with hybrid memory architectures  

---

# ✅ **10. Prefill/Decode (PD) Disaggregation (vLLM, SGLang)**

PD splits inference into two separate distributed systems:

```mermaid
flowchart LR
    A["Prefill Workers"] --> B["Prefill Outputs (KV+Hidden)"]
    B --> C["Decode Workers"]
```

### **Prefill Workers**
- Handle heavy GEMM throughput
- Batch many requests

### **Decode Workers**
- Handle autoregressive loop
- Use cached K/V from prefill

Benefits:
✅ Multi-host parallelism  
✅ Better GPU utilization  
✅ Improved tail latency  
✅ Great for long‑context workloads  

---

# ✅ **Final Summary**

| Technique | Purpose | Key Benefit |
|----------|----------|--------------|
| Static KV | Preallocated KV buffers | Simple but wasteful |
| Paged KV | KV blocks + page tables | Efficient multi-tenant inference |
| PagedAttention | Page-aware kernels | High throughput |
| LMCache | Multi-level KV (GPU/CPU/NVMe) | Long-context expansion |
| Parameter & Activation Disaggregation | Split model/KV across memory tiers | Run ultra-large models |
| Prefill/Decode Disaggregation | Split inference workflow | Higher throughput & concurrency |

---
