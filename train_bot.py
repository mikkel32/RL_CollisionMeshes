# ==============================================================================
# SOTA ROCKET LEAGUE AI - SIM-TO-REAL IMMORTAL ENGINE (SOTA V31)
# 40-Core EPYC / Automated Recovery Hook / Anti-Donut State Perfect Replica
# ==============================================================================

# 🛑 AUTO-DEPENDENCY INJECTION FOR GOOGLE COLAB 🛑
import sys
import subprocess
try:
    import onnxscript
except ImportError:
    print("📦 Installing missing 'onnxscript' & 'onnx' for Google Colab ONNX exports...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxscript", "-q"])

import os
import re
import math
import random
import warnings
import traceback
import json
import shutil
import logging
from collections import deque
from typing import Any

# 🛑 SILENCE ANNOYING PYTORCH & PYTHON WARNINGS 🛑
warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
logging.getLogger("torch.export").setLevel(logging.ERROR)

# 🛑 CRITICAL FIX 1: KILL "THREAD BOMB" (CPU Contention Fix) 🛑
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import gym
from tqdm import tqdm

import rlgym_sim
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.action_parsers import ActionParser
from rlgym_sim.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_sim.utils.state_setters import StateSetter, StateWrapper, DefaultState
from rlgym_ppo import Learner

from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityBallToGoalReward

torch.set_num_threads(1)

# ------------------------------------------------------------------------------
# 0. ALGEBRAIC INVERSE CONSTANTS (CPU Math Optimization)
# ------------------------------------------------------------------------------
INV_2044 = 1.0 / 2044.0
INV_2300 = 1.0 / 2300.0
INV_4096 = 1.0 / 4096.0
INV_5120 = 1.0 / 5120.0
INV_5_5  = 1.0 / 5.5
INV_6000 = 1.0 / 6000.0
INV_4600 = 1.0 / 4600.0
INV_10240= 1.0 / 10240.0
INV_120  = 8.0 / 120.0
INV_1_75 = 1.0 / 1.75
INV_3000 = 1.0 / 3000.0

# Hitbox normalization constants
INV_150 = 1.0 / 150.0  
INV_100 = 1.0 / 100.0  
INV_50  = 1.0 / 50.0   

# ------------------------------------------------------------------------------
# 1. DOMAIN RANDOMIZATION WRAPPERS (Sim-to-Real Protection)
# ------------------------------------------------------------------------------
class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, min_delay=0, max_delay=2):
        super().__init__(env)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.action_buffer = []

    def reset(self, **kwargs):
        self.action_buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        # ⚡ Fast NumPy Copy instead of deepcopy to maintain 150k SPS!
        self.action_buffer.append(np.array(action, copy=True)) 
        delay_ticks = random.randint(self.min_delay, self.max_delay)
        
        if len(self.action_buffer) > delay_ticks:
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = self.action_buffer[0] 
            
        return self.env.step(delayed_action)

class PhysicsRandomizationMutator(StateSetter):
    def __init__(self, base_mutator):
        super().__init__()
        self.base_mutator = base_mutator

    def reset(self, wrapper: StateWrapper):
        self.base_mutator.reset(wrapper)
        for car in wrapper.cars:
            vel = car.linear_velocity
            car.set_lin_vel(
                vel[0] * random.uniform(0.98, 1.02),
                vel[1] * random.uniform(0.98, 1.02),
                vel[2] * random.uniform(0.98, 1.02)
            )

def revert_collision_meshes():
    search_dirs = [".", "collision_meshes", "/content/RL_CollisionMeshes"]
    try: search_dirs.append(os.path.dirname(rlgym_sim.__file__))
    except NameError: pass
    for d in search_dirs:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for filename in files:
                match = re.match(r"^mesh_0(\d)\.cmf$", filename)
                if match:
                    try: os.rename(os.path.join(root, filename), os.path.join(root, f"mesh_{match.group(1)}.cmf"))
                    except OSError: pass

class RLBotONNXWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.net = getattr(policy, "model", policy)
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------
# 2. VECTORIZED ACTION PARSER 
# ------------------------------------------------------------------------------
class SOTAActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = np.array(self._make_bins(), dtype=np.float32)

    def _make_bins(self):
        bins = []
        for throttle in [-1.0, 0.0, 1.0]:
            for steer in [-1.0, 0.0, 1.0]:
                for pitch in [-1.0, 0.0, 1.0]:
                    for yaw in [-1.0, 0.0, 1.0]:
                        for roll in [-1.0, 0.0, 1.0]:
                            for jump in [0.0, 1.0]:
                                for boost in [0.0, 1.0]:
                                    for handbrake in [0.0, 1.0]:
                                        if boost == 1 and throttle == -1: continue 
                                        if steer != 0 and yaw != 0 and steer != yaw: continue
                                        bins.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
        return bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int32).flatten()
        actions = np.clip(actions, 0, len(self._lookup_table) - 1)
        parsed = self._lookup_table[actions].copy()
        
        for i, player in enumerate(state.players):
            if not player.on_ground and not player.has_flip:
                parsed[i, 5] = 0.0  
        return parsed

# ------------------------------------------------------------------------------
# 3. ZERO-ALLOCATION OBSERVATION BUILDER
# ------------------------------------------------------------------------------
class TemporalMemoryObservation(ObsBuilder):
    def __init__(self, action_parser: ActionParser, history_size=3):
        super().__init__()
        self.action_parser = action_parser
        self.history_size = history_size
        self.memory_banks = {}

    def reset(self, initial_state: GameState): 
        self.memory_banks.clear()

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        if player.team_num == 1: 
            car, ball, pads = player.inverted_car_data, state.inverted_ball, state.inverted_boost_pads
        else: 
            car, ball, pads = player.car_data, state.ball, state.boost_pads

        px, py, pz = car.position
        vx, vy, vz = car.linear_velocity
        ax, ay, az = car.angular_velocity
        bx, by, bz = ball.position
        bvx, bvy, bvz = ball.linear_velocity
        fx, fy, fz = car.forward()
        ux, uy, uz = car.up()
        rx, ry, rz = (fy*uz - fz*uy), (fz*ux - fx*uz), (fx*uy - fy*ux)

        h_len = player.car_data.hitbox_size[0] if hasattr(player.car_data, 'hitbox_size') else 118.01 
        h_wid = player.car_data.hitbox_size[1] if hasattr(player.car_data, 'hitbox_size') else 84.20
        h_hei = player.car_data.hitbox_size[2] if hasattr(player.car_data, 'hitbox_size') else 36.16

        obs = [
            px * INV_4096, py * INV_5120, pz * INV_2044, 
            vx * INV_2300, vy * INV_2300, vz * INV_2300,
            (vx*fx + vy*fy + vz*fz) * INV_2300,
            (vx*rx + vy*ry + vz*rz) * INV_2300,
            (vx*ux + vy*uy + vz*uz) * INV_2300,
            (ax*fx + ay*fy + az*fz) * INV_5_5,
            (ax*rx + ay*ry + az*rz) * INV_5_5,
            (ax*ux + ay*uy + az*uz) * INV_5_5,
            fx, fy, fz, rx, ry, rz, ux, uy, uz,
            (bx - px) * INV_10240, (by - py) * INV_10240, (bz - pz) * INV_2044, 
            (bvx - vx) * INV_6000, (bvy - vy) * INV_6000, (bvz - vz) * INV_6000, 
            
            math.sqrt(max(0.0, player.boost_amount / 100.0 if player.boost_amount > 1.0 else player.boost_amount)),
            float(player.on_ground), float(player.has_flip), float(player.is_demoed),
            
            h_len * INV_150, h_wid * INV_100, h_hei * INV_50
        ]

        obs.extend(pads.tolist())

        found_opp = False
        for other in state.players:
            if other.car_id != player.car_id:
                o_car = other.inverted_car_data if player.team_num == 1 else other.car_data
                ox, oy, oz = o_car.position
                ovx, ovy, ovz = o_car.linear_velocity
                obs.extend([
                    (ox - px) * INV_10240, (oy - py) * INV_10240, (oz - pz) * INV_2044,
                    (ovx - vx) * INV_4600, (ovy - vy) * INV_4600, (ovz - vz) * INV_4600
                ])
                found_opp = True
                break 
        
        if not found_opp:
            obs.extend([0.0] * 6)

        try:
            prev_act = np.asarray(previous_action)
            if prev_act.size == 8:
                obs.extend(prev_act.flatten().tolist())
            elif prev_act.size == 1:
                idx = int(prev_act.item())
                idx = max(0, min(idx, len(self.action_parser._lookup_table) - 1)) 
                obs.extend(self.action_parser._lookup_table[idx].tolist())
            else:
                obs.extend([0.0] * 8)
        except Exception:
            obs.extend([0.0] * 8)

        cid = player.car_id
        if cid not in self.memory_banks:
            self.memory_banks[cid] = deque([obs] * self.history_size, maxlen=self.history_size)
        else:
            self.memory_banks[cid].append(obs)
            
        flat_obs = [val for frame in self.memory_banks[cid] for val in frame]
        
        obs_arr = np.array(flat_obs, dtype=np.float32)
        if not np.isfinite(obs_arr).all():
            obs_arr = np.nan_to_num(obs_arr)
            
        return obs_arr

# ------------------------------------------------------------------------------
# 4. ALGEBRAICALLY SIMPLIFIED REWARD SHAPING
# ------------------------------------------------------------------------------
class CompoundAerialReward(RewardFunction):
    def __init__(self): self.air_time = {}
    def reset(self, initial_state: GameState): self.air_time.clear()
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cid = player.car_id
        if not player.on_ground: self.air_time[cid] = self.air_time.get(cid, 0.0) + INV_120
        else: self.air_time[cid] = 0.0
        rew = min(min(self.air_time.get(cid, 0.0) * INV_1_75, 1.0), min(max(player.car_data.position[2], 0.0) * INV_2044, 1.0))
        return float(rew) if not math.isnan(rew) else 0.0

class DynamicTouchReward(RewardFunction):
    def __init__(self): self.last_vel = {}
    def reset(self, initial_state: GameState): self.last_vel.clear()
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        rew = 0.0
        cid = player.car_id
        if player.ball_touched and cid in self.last_vel:
            bx, by, bz = state.ball.linear_velocity
            lx, ly, lz = self.last_vel[cid]
            vel_delta = math.sqrt((bx-lx)**2 + (by-ly)**2 + (bz-lz)**2)
            rew = (vel_delta * INV_6000) * max(player.car_data.position[2] * INV_2044, 0.1)
        self.last_vel[cid] = tuple(state.ball.linear_velocity)
        return float(rew) if not math.isnan(rew) else 0.0

class VectorAlignmentReward(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        vx, vy, vz = player.car_data.linear_velocity
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        if speed < 10: return 0.0
        px, py, pz = player.car_data.position
        bx, by, bz = state.ball.position
        dx, dy, dz = bx-px, by-py, bz-pz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 10: 
            rew = (vx*dx + vy*dy + vz*dz) / (speed * dist)
            return float(rew) if not math.isnan(rew) else 0.0
        return 0.0

class KinestheticShadowDefense(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cx, cy, cz = player.car_data.position
        bx, by, bz = state.ball.position
        dist = math.sqrt((cx-bx)**2 + (cy-by)**2 + (cz-bz)**2)
        dist_factor = math.exp(-dist * INV_3000)
        gy = -5120.0 if player.team_num == 0 else 5120.0
        b2gx, b2gy, b2gz = -bx, gy-by, -bz
        c2gx, c2gy, c2gz = -cx, gy-cy, -cz
        b2g_n = math.sqrt(b2gx**2 + b2gy**2 + b2gz**2)
        c2g_n = math.sqrt(c2gx**2 + c2gy**2 + c2gz**2)
        align = 0.0
        if b2g_n > 0 and c2g_n > 0: align = (b2gx*c2gx + b2gy*c2gy + b2gz*c2gz) / (b2g_n * c2g_n)
        bvx, bvy, bvz = state.ball.linear_velocity
        pvx, pvy, pvz = player.car_data.linear_velocity
        bv_n = math.sqrt(bvx**2 + bvy**2 + bvz**2)
        pv_n = math.sqrt(pvx**2 + pvy**2 + pvz**2)
        v_match = 0.0
        if bv_n > 0 and pv_n > 0: v_match = (bvx*pvx + bvy*pvy + bvz*pvz) / (bv_n * pv_n)
        rew = float(dist_factor * max(0.0, align) * max(0.0, v_match))
        return rew if not math.isnan(rew) else 0.0

class BackwardVelocityPenalty(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        vx, vy, vz = player.car_data.linear_velocity
        fx, fy, fz = player.car_data.forward()
        forward_vel = vx*fx + vy*fy + vz*fz
        if forward_vel < -150:
            rew = -float(abs(forward_vel) * INV_2300)
            return rew if not math.isnan(rew) else 0.0
        return 0.0

class BoostDifferenceReward(RewardFunction):
    def __init__(self): self.last_boost = {}
    def reset(self, initial_state: GameState): self.last_boost.clear()
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cid = player.car_id
        current_boost = player.boost_amount / 100.0 if player.boost_amount > 1.0 else player.boost_amount
        last_boost = self.last_boost.get(cid, current_boost)
        self.last_boost[cid] = current_boost
        if current_boost > last_boost: 
            return float(math.sqrt(current_boost) - math.sqrt(last_boost))
        return 0.0

# ------------------------------------------------------------------------------
# 5. CURRICULUM MUTATORS
# ------------------------------------------------------------------------------
class EscalateMutator(StateSetter):
    def reset(self, wrapper: StateWrapper):
        scenario = random.random()
        
        if scenario < 0.25:
            wrapper.ball.set_pos(0.0, 0.0, random.uniform(1200, 1800))
            wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
            for car in wrapper.cars:
                y_pos = -2000.0 if car.team_num == 0 else 2000.0
                yaw = math.pi / 2 if car.team_num == 0 else -math.pi / 2
                car.set_pos(0.0, y_pos, 17.0)
                car.set_rot(0.0, yaw, 0.0)
                car.set_lin_vel(0.0, 0.0, 0.0)
                car.boost = 1.0
                
        elif scenario < 0.50:
            side = random.choice([-1.0, 1.0])
            for car in wrapper.cars:
                y_pos = -500.0 if car.team_num == 0 else 500.0
                car.set_pos(3000.0 * side, y_pos, 200.0)
                car.set_rot(0.0, 0.0, (math.pi/2) * side)
                car.set_lin_vel(0.0, 1500.0, 600.0)
                car.boost = 1.0
            wrapper.ball.set_pos(3000.0 * side, 100.0, 900.0)
            wrapper.ball.set_lin_vel(0.0, 1500.0, 600.0)
            
        elif scenario < 0.75:
            wrapper.ball.set_pos(0.0, 0.0, 100.0)
            wrapper.ball.set_lin_vel(0.0, 0.0, 0.0)
            for car in wrapper.cars:
                car.set_pos(random.uniform(-3000, 3000), random.uniform(-4000, 4000), random.uniform(500, 1500))
                car.set_rot(random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))
                car.set_ang_vel(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
                car.boost = random.uniform(0.0, 0.5)
                
        else:
            DefaultState().reset(wrapper)

# ------------------------------------------------------------------------------
# 6. ENVIRONMENT GENERATION
# ------------------------------------------------------------------------------
def build_env():
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)

    reward_fn = CombinedReward(
        (
            EventReward(goal=100.0, concede=-100.0), 
            VelocityBallToGoalReward(),            
            CompoundAerialReward(),                
            DynamicTouchReward(),                  
            VectorAlignmentReward(),               
            KinestheticShadowDefense(),            
            BackwardVelocityPenalty(),             
            BoostDifferenceReward()                
        ),
        (1.0, 0.05, 0.005, 0.02, 0.001, 0.01, 0.02, 0.05) 
    )
    
    action_parser = SOTAActionParser()
    robust_state_setter = PhysicsRandomizationMutator(EscalateMutator())
    
    env = rlgym_sim.make(
        tick_skip=8, team_size=1, spawn_opponents=True,
        reward_fn=reward_fn, 
        obs_builder=TemporalMemoryObservation(action_parser=action_parser, history_size=3),
        action_parser=action_parser, 
        state_setter=robust_state_setter,
        terminal_conditions=[TimeoutCondition(400), GoalScoredCondition()]
    )
    
    env = ActionDelayWrapper(env, min_delay=0, max_delay=2)
    return env

# ------------------------------------------------------------------------------
# 7. SOTA V31 MAIN PPO ENGINE (AUTOMATED RECOVERY ENGINE)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        pass
        
    revert_collision_meshes()

    print("🚀 Initializing THE SIM-TO-REAL INFINITE ENGINE (V31)...")
    
    try:
        temp_env = build_env()
        dummy_reset = temp_env.reset()
        if isinstance(dummy_reset, tuple): dummy_reset = dummy_reset[0]
        obs_size = np.atleast_2d(dummy_reset).shape[-1]
        temp_env.close()
        print(f"✅ Domain Randomization Env Built! True Obs Size: {obs_size}")
    except Exception as e:
        print(f"🚨 FATAL: build_env() crashed!\n{traceback.format_exc()}")
        sys.exit(1)

    WORKER_CORES = 40 
    GLOBAL_BATCH_SIZE = 300_000 
    MINI_BATCH = 150_000 
    
    BASE_ITERS = 2000
    EXTENSION_STEP = 1000
    TOTAL_ITERS = BASE_ITERS
    
    learner = Learner(
        build_env,
        n_proc=WORKER_CORES, 
        ppo_batch_size=GLOBAL_BATCH_SIZE,
        ts_per_iteration=GLOBAL_BATCH_SIZE,
        exp_buffer_size=GLOBAL_BATCH_SIZE, 
        ppo_minibatch_size=MINI_BATCH, 
        ppo_ent_coef=0.01,
        
        policy_lr=2e-4,
        critic_lr=4e-4, 
        ppo_epochs=10, 
        
        policy_layer_sizes=(512, 512, 512),               
        critic_layer_sizes=(4096, 4096, 2048, 1024),      
        
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_to_wandb=False
    )

    # 🛑 ♻️ THE ULTIMATE AUTO-RESUME PROTOCOL WITH RECOVERY HOOK ♻️ 🛑
    start_iter = 0
    ckpt_dir = "/content/drive/MyDrive/RocketLeagueModel/Checkpoints"
    
    if os.path.exists(ckpt_dir):
        print(f"\n🔍 Scanning {ckpt_dir} for previous saves...")
        
        all_files_and_dirs = os.listdir(ckpt_dir)
        valid_iters = []
        for f in all_files_and_dirs:
            match = re.search(r'(?:ckpt_V\d+_|ckpt_|raw_policy_weights_)(\d+)', f)
            if match:
                valid_iters.append(int(match.group(1)))

        if valid_iters:
            start_iter = max(valid_iters)
            print(f"🔄 FOUND EXISTING CLOUD SAVE! Highest iteration detected: {start_iter}")
            
            while start_iter >= TOTAL_ITERS:
                TOTAL_ITERS += EXTENSION_STEP
                print(f"📈 Cap Reached! Automatically extending training horizon to {TOTAL_ITERS} iterations.")

            possible_ckpt_names = [f"ckpt_V31_{start_iter}", f"ckpt_V30_{start_iter}", f"ckpt_V27_{start_iter}", f"ckpt_{start_iter}"]
            ckpt_path = None
            for name in possible_ckpt_names:
                if os.path.exists(os.path.join(ckpt_dir, name)):
                    ckpt_path = os.path.join(ckpt_dir, name)
                    break
                    
            # 🛑 THE AUTOMATED RECOVERY HOOK (Fixes the Panicked Donut State)
            if ckpt_path and os.path.exists(ckpt_path):
                # We actively hunt for missing critical memory files and pull them from the past
                for req_file in ["BOOK_KEEPING_VARS.json", "OBS_STANDARDIZER.pt", "REWARD_STANDARDIZER.pt", "config.json"]:
                    tgt_path = os.path.join(ckpt_path, req_file)
                    if not os.path.exists(tgt_path):
                        if req_file == "config.json":
                            with open(tgt_path, "w") as f: json.dump({}, f)
                            continue
                            
                        print(f"⚠️ Missing {req_file} in {ckpt_path}! Initiating Auto-Recovery...")
                        for f in sorted(all_files_and_dirs, reverse=True):
                            if f.startswith("ckpt_") and f != os.path.basename(ckpt_path):
                                legacy_file = os.path.join(ckpt_dir, f, req_file)
                                if os.path.exists(legacy_file):
                                    try:
                                        shutil.copyfile(legacy_file, tgt_path)
                                        print(f"   ✅ SUCCESS: Dragged missing {req_file} from {f} -> {ckpt_path}")
                                        break
                                    except Exception:
                                        pass

            raw_pt_path = os.path.join(ckpt_dir, f"raw_policy_weights_{start_iter}.pt")
            loaded = False
            
            try:
                try: policy_net = learner.ppo_learner.policy
                except AttributeError: policy_net = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
                device = next(policy_net.parameters()).device

                # Layer 1: NATIVE LOAD 
                if ckpt_path and os.path.exists(os.path.join(ckpt_path, "PPO_POLICY.pt")):
                    try:
                        learner.load(ckpt_path)
                        print(f"   ✅ NATIVE LOAD SUCCESS: Loaded full PyTorch brain & bookkeeping from {ckpt_path}")
                        loaded = True
                    except Exception as e:
                        print(f"   ⚠️ Native load failed: {e}. Attempting manual injection...")

                        try:
                            policy_net.load_state_dict(torch.load(os.path.join(ckpt_path, "PPO_POLICY.pt"), map_location=device), strict=False)
                            try: learner.ppo_learner.value_net.load_state_dict(torch.load(os.path.join(ckpt_path, "PPO_VALUE_NET.pt"), map_location=device), strict=False)
                            except: pass
                            
                            if os.path.exists(os.path.join(ckpt_path, "OBS_STANDARDIZER.pt")) and hasattr(learner.ppo_learner, "obs_standardizer"):
                                try: learner.ppo_learner.obs_standardizer.load_state_dict(torch.load(os.path.join(ckpt_path, "OBS_STANDARDIZER.pt"), map_location=device))
                                except: pass

                            bk_path = os.path.join(ckpt_path, "BOOK_KEEPING_VARS.json")
                            if os.path.exists(bk_path):
                                with open(bk_path, 'r') as f:
                                    bk_vars = json.load(f)
                                    if "cumulative_timesteps" in bk_vars:
                                        learner.agent.cumulative_timesteps = bk_vars["cumulative_timesteps"]
                                        
                            print(f"   ✅ Manually Restored PyTorch Brain & Standardizers from folder.")
                            loaded = True
                        except Exception as e_man:
                            print(f"   ⚠️ Manual load failed: {e_man}")

                # Ultimate Fallback: The Raw Actor Weights
                if not loaded and os.path.exists(raw_pt_path):
                    try:
                        policy_net.load_state_dict(torch.load(raw_pt_path, map_location=device), strict=False)
                        print(f"   ✅ Restored Neural Network Actor Brain from {raw_pt_path} (Warning: Standardizers reset to 0)")
                        loaded = True
                    except Exception as e:
                        print(f"   ❌ Failed to load raw weights: {e}")
                        
                if loaded:
                    try: learner.agent.cumulative_timesteps = max(learner.agent.cumulative_timesteps, start_iter * GLOBAL_BATCH_SIZE)
                    except: pass
                    print(f"🚀 Continuing training seamlessly from Iteration {start_iter}...\n")
                else:
                    start_iter = 0
                    TOTAL_ITERS = BASE_ITERS
                    print("⚠️ Found files but failed to restore. Starting fresh...\n")
            except Exception as e:
                print(f"⚠️ Initialization error during restore: {e}")
                start_iter = 0
                TOTAL_ITERS = BASE_ITERS

    try:
        for i in tqdm(range(start_iter, TOTAL_ITERS), desc=f"Training GC Bot ({TOTAL_ITERS} Iters)", initial=start_iter, total=TOTAL_ITERS, file=sys.stdout):
            
            experience, metrics, steps, coll_time = learner.agent.collect_timesteps(GLOBAL_BATCH_SIZE)
            
            learner.add_new_experience(experience)
            learner.ppo_learner.learn(learner.experience_buffer)
            learner.agent.cumulative_timesteps += steps
            
            progress = (i + 1) / TOTAL_ITERS
            new_policy_lr = 2e-4 - ((2e-4 - 1e-5) * progress)
            new_critic_lr = 4e-4 - ((4e-4 - 5e-5) * progress) 
            new_ent = 0.01 - ((0.01 - 0.005) * progress)
            
            try:
                for param_group in learner.ppo_learner.policy_optimizer.param_groups: param_group['lr'] = new_policy_lr
                for param_group in learner.ppo_learner.value_optimizer.param_groups: param_group['lr'] = new_critic_lr
            except AttributeError:
                pass
            
            learner.ppo_ent_coef = new_ent
            learner.ppo_learner.ent_coef = new_ent

            # 🛑 DIRECT PYTORCH CHECKPOINTING WITH PERFECT REPLICA (Saves STRICTLY at 500, 1000...)
            if (i + 1) > start_iter and (i + 1) % 500 == 0:
                print(f"\n💾 Initiating Cloud Backup for Iteration {i+1}...")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                ckpt_folder = os.path.join(ckpt_dir, f"ckpt_V31_{i+1}")
                os.makedirs(ckpt_folder, exist_ok=True)
                
                try:
                    # Natively extract all neural network and standardizer files without the buggy shutil.copy
                    learner.ppo_learner.save_to(ckpt_folder)
                    
                    # Manually construct the bookkeeping JSON
                    bk_vars = {"cumulative_timesteps": int(learner.agent.cumulative_timesteps)}
                    with open(os.path.join(ckpt_folder, "BOOK_KEEPING_VARS.json"), "w") as f:
                        json.dump(bk_vars, f)
                        
                    # Dummy config to keep learner.load() happy for future runs
                    with open(os.path.join(ckpt_folder, "config.json"), "w") as f:
                        json.dump({}, f)
                        
                    print(f"   ✅ Perfect Replica Checkpoint Secure: All files saved to {ckpt_folder}!")
                except Exception as e:
                    print(f"   ⚠️ Perfect Replica save failed: {e}")
                
                # Raw Fallback & ONNX Export
                try:
                    try: policy_net = learner.ppo_learner.policy
                    except AttributeError: policy_net = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
                    device_net = next(policy_net.parameters()).device
                    
                    fallback_path = os.path.join(ckpt_dir, f"raw_policy_weights_{i+1}.pt")    
                    torch.save(policy_net.state_dict(), fallback_path)
                    
                    onnx_path = os.path.join(ckpt_dir, f"SOTA_RLBot_V31_Iter_{i+1}.onnx")
                    dummy_in = torch.randn(1, obs_size, dtype=torch.float32, device=device_net)
                    onnx_safe_policy = RLBotONNXWrapper(policy_net).eval()
                    
                    torch.onnx.export(
                        onnx_safe_policy, dummy_in, onnx_path,
                        export_params=True, opset_version=18, do_constant_folding=True,
                        input_names=['observation'], output_names=['action_logits'],
                        dynamic_axes={'observation': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
                    )
                    policy_net.train() 
                    print(f"   ✅ ONNX HOT-SWAP EXPORT: Dynamic-Batched model saved to Drive.")
                except Exception as e_pt:
                    print(f"   ❌ FATAL: Override Backup Failed: {e_pt}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted safely.")
    except Exception as e:
        print(f"\n🚨 CRASH DURING TRAINING:\n{traceback.format_exc()}")
    finally:
        learner.cleanup()

    print("\n🔥 Training Concluded! Quantizing final ACTOR ONLY to ONNX...")
    
    try:
        try: policy_net = learner.ppo_learner.policy
        except AttributeError: policy_net = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
            
        policy_net.to("cpu")
        onnx_safe_policy = RLBotONNXWrapper(policy_net).eval()
        dummy_input = torch.randn(1, obs_size, dtype=torch.float32, device="cpu")
        
        save_dir = "/content/drive/MyDrive/RocketLeagueModel"
        export_path_drive = os.path.join(save_dir, "SOTA_RLBot_V31_Final.onnx")
        export_path_fallback = "SOTA_RLBot_V31_FALLBACK.onnx"
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            torch.onnx.export(
                onnx_safe_policy, dummy_input, export_path_drive,
                export_params=True, opset_version=18, do_constant_folding=True,
                input_names=['observation'], output_names=['action_logits'],
                dynamic_axes={'observation': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
            )
            print(f"✅ FINAL ACTOR WEIGHTS EXPORTED SAFELY TO GOOGLE DRIVE -> {export_path_drive}")
            
        except Exception as e_drive:
            print(f"\n⚠️ WARNING: Google Drive export failed! (Did the drive unmount?)")
            print("🔄 Executing Local Colab Backup Save...")
            try:
                torch.onnx.export(
                    onnx_safe_policy, dummy_input, export_path_fallback,
                    export_params=True, opset_version=18, do_constant_folding=True,
                    input_names=['observation'], output_names=['action_logits'],
                    dynamic_axes={'observation': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
                )
                print(f"✅ CRISIS AVERTED: Weights saved locally -> {export_path_fallback}")
            except Exception as e_local:
                pass
    except Exception as e_final:
        pass
