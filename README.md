# HackWestern12
# 🧠 Crowd Density Detection — Soft-CSRNet / CSRNet

This project connects to a drone or local video feed to estimate **crowd density** and render live heatmaps using **Soft-CSRNet** (default) or **CSRNet** (official architecture).  
Ideal for UAV-based monitoring, smart city analytics, or live event analysis.

---

## 🚀 Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/yourusername/HackWestern12.git
cd HackWestern12

python -m venv .venv
. .venv/Scripts/Activate.ps1   # Windows PowerShell
# or
source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt

# run density thing
python -m src.app_softcsrnet_local --video "data/crowd2.mp4" --weights "models/PartAmodel_best.pth"

