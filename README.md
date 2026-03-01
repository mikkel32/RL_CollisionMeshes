<div align="center">

# 🏎️ RLGym & RocketSim Bot Training Environment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![RLGym-PPO](https://img.shields.io/badge/RLGym--PPO-Latest-success)](https://github.com/AechPro/rlgym-ppo)

*A streamlined and automated Google Colab setup for training Rocket League bots using RLGym-PPO and RocketSim.*

</div>

---

## 📖 Overview

Welcome to the Rocket League bot training setup! This guide provides the exact steps needed to configure your Google Colab environment, resolve dependency conflicts, and start training your bot. 

Because the physics simulation relies on specific C++ headers and 3D collision meshes, the setup process requires a specific execution order outlined below.

---

## ⚠️ CRITICAL: First-Run Restart Required

> [!WARNING]  
> **A session restart is REQUIRED on your first run.**  
> Google Colab comes with NumPy 2.x pre-installed, which breaks the C++ headers required by the RocketSim environment. We must aggressively downgrade it to NumPy 1.x.
> 
> Because we are modifying core system libraries in use, **an error will appear on your first run**. This is completely normal and expected!
> 
> **How to fix:**
> 1. Run **Cell 1** and let the installation finish (ignore the error prompt at the end).
> 2. Go to the top menu and click **`Runtime`** ➔ **`Restart session`** (or press `Ctrl+M` + `.`).
> 3. **Re-run Cell 1**. The error will disappear, and the environment will be fully ready!

---

## ⚙️ Step 1: Environment & Hardware Setup

This initial cell prepares your hardware, resolves C-API dependencies, installs the RLGym ecosystem directly from the source repositories, and downloads the exact collision meshes.

Copy and paste the following code into **Cell 1** of your Colab notebook:

```python
# ==========================================
# CELL 1: HARDWARE SETUP & NUMPY C-API FIX
# ==========================================

# 1. 🚨 CRITICAL: You MUST downgrade numpy to 1.x! NumPy 2.x breaks the C++ headers.
!pip install -q "numpy<2.0.0" gym

# 2. Install the rest of the ecosystem (Fetches rlgym-ppo directly from Git)
!pip install -q git+[https://github.com/AechPro/rlgym-ppo.git](https://github.com/AechPro/rlgym-ppo.git)
!pip install -q rocketsim torch onnx wandb tabulate
!pip install -q git+[https://github.com/AechPro/rocket-league-gym-sim@main](https://github.com/AechPro/rocket-league-gym-sim@main)

# 3. Clone the exact collision meshes (Removes existing directories to prevent Git conflicts)
!rm -rf RL_CollisionMeshes collision_meshes
!git clone [https://github.com/mikkel32/RL_CollisionMeshes.git](https://github.com/mikkel32/RL_CollisionMeshes.git)
!mkdir -p collision_meshes
!cp -r RL_CollisionMeshes/collision_meshes/soccar/* collision_meshes/

print("✅ System Ready. YOU MUST RESTART THE COLAB SESSION NOW IF THIS WAS YOUR FIRST RUN.")
