Cell 1:
Quick Note, It will show an error on first run, meaning it needs to restart the session, after u restart, rerun the cell, the error will dissapear.

# ==========================================
# CELL 1: HARDWARE SETUP & NUMPY C-API FIX
# ==========================================
# 1. 🚨 CRITICAL: You MUST downgrade numpy to 1.x! NumPy 2.x breaks the C++ headers.
!pip install -q "numpy<2.0.0" gym

# 2. Install the rest of the ecosystem (Henter rlgym-ppo direkte fra Git)
!pip install -q git+https://github.com/AechPro/rlgym-ppo.git
!pip install -q rocketsim torch onnx wandb tabulate
!pip install -q git+https://github.com/AechPro/rocket-league-gym-sim@main

# 3. Clone the exact collision meshes (Sletter gamle spor for at undgå Git-fejl)
!rm -rf RL_CollisionMeshes collision_meshes
!git clone https://github.com/mikkel32/RL_CollisionMeshes.git
!mkdir -p collision_meshes
!cp -r RL_CollisionMeshes/collision_meshes/soccar/* collision_meshes/
!echo "✅ System Ready. YOU MUST RESTART THE COLAB SESSION NOW."


Cell 2:

%cd /content/RL_CollisionMeshes/
!python train_bot.py
