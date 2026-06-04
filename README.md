# Markov Regime-Switching Model for Options Pricing

Code sample submitted in support of application RAD_NYC_DAT_6444 — Data Analyst, Research and Analytics Department, New York State Office of the Attorney General.

**[Portfolio Page](https://barcogregory.com/oag_application_sample)** | **[Methodology Overview](https://barcogregory.com/static/OAG_APP_Writing_Sample_Methodology_Overview.pdf)**

---

## [A] Python ETL Pipeline
`auth.py` `market_data.py` `contract_selector.py`

Authentication, token lifecycle management, options chain extraction, contract filtering and selection. These three files form the full data ingestion layer from API call to structured output ready for downstream analysis.

## [B] Data Visualization
`landscape_scan.py`

Interactive 3D fitness landscape rendering across 6 parameter pairs. WebGL-accelerated Plotly surfaces exported to self-contained HTML with hover coordinates, annotated seed markers, and dropdown pair switching. No server dependency.

## Development Environment

Ubuntu 26.04 WSL2 with NVIDIA HPC SDK (nvfortran), Intel MKL, and OpenMP -- enabling CUDA GPU parallel execution on an RTX 5060 Ti achieving approximately 100x speedup over standard CPU computation. Python orchestration interfaces with a compiled Fortran numerical library via f2py bindings.

---

*Best, G. Barco*
