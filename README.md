<div align="center">

# 🏎️ RLGym & RocketSim Colab Training Environment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![RLGym-PPO](https://img.shields.io/badge/RLGym--PPO-Latest-success)](https://github.com/AechPro/rlgym-ppo)

*A streamlined, professional Google Colab setup for training Rocket League bots using RLGym-PPO and RocketSim.*

</div>

---

## 📖 Overview

Welcome to the Rocket League bot training repository! This guide provides the exact, step-by-step instructions required to configure your Google Colab environment, resolve dependency conflicts, and successfully initiate your training loop.

Because the RocketSim physics simulation relies on heavily optimized C++ headers and specific 3D collision meshes, the installation requires a strict execution order. Please follow the instructions below carefully.

---

## ⚠️ CRITICAL: First-Run Restart Requirement

> [!WARNING]  
> **A session restart is REQUIRED on your very first run.**  
> Google Colab comes with NumPy 2.x pre-installed, which breaks the C++ headers required by the RocketSim physics engine. We must forcefully downgrade it to NumPy 1.x.
>
> Because we are overwriting core system libraries that Colab is actively using, **a red error message will appear when Cell 1 finishes**. This is completely normal and expected! The step-by-step guide below explains exactly how to handle it.

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Enable GPU Acceleration (Recommended)
Before running any code, ensure your Colab notebook is utilizing a GPU to vastly accelerate your training.
1. In the top menu, click **`Runtime`** ➔ **`Change runtime type`**.
2. Set the **Hardware accelerator** to **`T4 GPU`** (or better) and click **Save**.

### Step 2: Hardware & Dependency Setup (Cell 1)
Create a new code cell in your Colab notebook (**Cell 1**). Copy and paste the code below into it. This cell downgrades NumPy, installs the complete RLGym ecosystem directly from the source repositories, and prepares the collision meshes.

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

# 3. Clone the exact collision meshes (Removes old files to prevent Git conflicts)
!rm -rf RL_CollisionMeshes collision_meshes
!git clone [https://github.com/mikkel32/RL_CollisionMeshes.git](https://github.com/mikkel32/RL_CollisionMeshes.git)
!mkdir -p collision_meshes
!cp -r RL_CollisionMeshes/collision_meshes/soccar/* collision_meshes/

!echo "✅ System Ready. YOU MUST RESTART THE COLAB SESSION NOW."
```

### Step 3: Restart the Session & Re-run!
1. **Run Cell 1** and let the installation finish. **Ignore the error prompt at the end.**
2. Go to the top menu in Colab: Click **`Runtime`** ➔ **`Restart session`** (or press `Ctrl+M` + `.`).
3. **Re-run Cell 1**. The error will vanish, and your environment is now perfectly configured and compiled!

### Step 4: Initiate Bot Training (Cell 2)
Now that the environment is stable, you can start the actual reinforcement learning process. Create a second code cell (**Cell 2**), paste the code below, and run it.

```python
# ==========================================
# CELL 2: EXECUTE BOT TRAINING
# ==========================================

# Navigate to the repository directory containing the training script
%cd /content/RL_CollisionMeshes/

# Start the training process
!python train_bot.py
```

---

## 💡 Troubleshooting & Notes

* **Why do we run `!rm -rf` in Cell 1?**  
  Running the remove command before cloning ensures that if you accidentally run the cell multiple times or reconnect to a runtime, Git won't crash with a `fatal: destination path already exists` error. It guarantees a fresh, clean install every time.

* **WandB Prompts?**  
  If your `train_bot.py` uses Weights & Biases (`wandb`) for tracking training metrics, Cell 2 might pause and ask for your API key. Simply paste your WandB API key into the Colab output box and press Enter.

* **Saving Progress:**  
  Remember that Google Colab wipes its storage when the session disconnects. Make sure your `train_bot.py` script is configured to save checkpoints to Google Drive or automatically sync to a cloud service like WandB.
