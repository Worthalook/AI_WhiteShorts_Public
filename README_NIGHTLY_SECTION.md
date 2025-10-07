## 🕓 Nightly Workflow (Recommended)

The GP_Refactor branch is now structured for a simple **three-step nightly pipeline**:

1. **Update** — fetch FINAL results from SportsData.io and merge into your historical union CSV  
2. **Train** — retrain RF + DSS models (weighted YTD + Last season)  
3. **Predict** — generate forecasts for the upcoming games via projections API  

---

### 🔧 Prerequisites

Install the package once:
```bash
pip install -e .
