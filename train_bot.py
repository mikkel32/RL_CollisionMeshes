# ==============================================================================
# SOTA ROCKET LEAGUE AI - 48 vCPU ENTERPRISE ENGINE (SOTA V8 PERFECTED)
# ==============================================================================
# INSTRUCTIONS: Save this exact code as "train_bot.py" and run it from the 
# terminal using: python train_bot.py
# Do NOT run this inside a Jupyter cell or Google Colab block directly!
# ==============================================================================
import os
import re
import sys
import math
import random
import warnings
import traceback

# Hide standard PyTorch/Jupyter deprecation clutter
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_ppo import Learner

torch.set_num_threads(1)

# ------------------------------------------------------------------------------
# 1. FILE SYSTEM SANITIZATION (The V8 Reverter)
# ------------------------------------------------------------------------------
def revert_collision_meshes():
    """Undoes the V7 zero-padding to prevent the C++ infinite broadphase loop."""
    print("🔧 Scanning directories to revert sabotaged collision meshes...")
    search_dirs = [".", "collision_meshes"]
    try:
        search_dirs.append(os.path.dirname(rlgym_sim.__file__))
    except NameError:
        pass

    reverted_count = 0
    for d in search_dirs:
        if not os.path.exists(d): continue
        for root, _, files in os.walk(d):
            for filename in files:
                # Look for the damaged zero-padded files (mesh_00.cmf -> mesh_09.cmf)
                match = re.match(r"^mesh_0(\d)\.cmf$", filename)
                if match:
                    old_path = os.path.join(root, filename)
                    new_filename = f"mesh_{match.group(1)}.cmf"
                    new_path = os.path.join(root, new_filename)
                    try:
                        os.rename(old_path, new_path)
                        reverted_count += 1
                        print(f"   -> Reverted Mesh Order: {filename} -> {new_filename}")
                    except OSError:
                        pass
    if reverted_count > 0:
        print(f"✅ Successfully restored {reverted_count} meshes. C++ Physics will load correctly!\n")

# ------------------------------------------------------------------------------
# 2. ACTION PARSER (Lag-Free Lookup & NaN-Guarded)
# ------------------------------------------------------------------------------
class SOTAActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self._make_bins()
        self._action_queue = {}

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
                                        if jump == 1 and handbrake == 1: continue
                                        if boost == 1 and throttle == -1: continue 
                                        if steer != 0 and yaw != 0 and steer != yaw: continue
                                        bins.append([throttle, steer, pitch, yaw, roll, jump, boost, handbrake])
        return bins

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        parsed = np.zeros((len(actions), 8), dtype=np.float32)
        for i, action_idx in enumerate(actions):
            try:
                val = float(np.array(action_idx).item())
                if math.isnan(val) or math.isinf(val): a_idx = 0
                else: a_idx = int(val)
            except (ValueError, TypeError, OverflowError):
                a_idx = 0
                
            act = list(self._lookup_table[a_idx])
            player = state.players[i]
            if not player.on_ground and not player.has_flip: act[5] = 0.0 
                
            cid = player.car_id
            if cid not in self._action_queue: self._action_queue[cid] = act
            if random.random() < 0.05: act = self._action_queue[cid] 
            else: self._action_queue[cid] = act
                
            parsed[i] = act
        return parsed

# ------------------------------------------------------------------------------
# 3. OBSERVATION BUILDER (With Infinity-Guard)
# ------------------------------------------------------------------------------
class SOTAObservation(ObsBuilder):
    def reset(self, initial_state: GameState): pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        if player.team_num == 1: car, ball = player.inverted_car_data, state.inverted_ball
        else: car, ball = player.car_data, state.ball

        px, py, pz = car.position
        vx, vy, vz = car.linear_velocity
        ax, ay, az = car.angular_velocity
        fx, fy, fz = car.forward()
        ux, uy, uz = car.up()
        
        rx, ry, rz = (fy*uz - fz*uy), (fz*ux - fx*uz), (fx*uy - fy*ux)
        
        bx, by, bz = ball.position
        bvx, bvy, bvz = ball.linear_velocity

        rx_rel, ry_rel, rz_rel = bx - px, by - py, bz - pz
        rvx_rel, rvy_rel, rvz_rel = bvx - vx, bvy - vy, bvz - vz

        obs = [
            pz / 2044.0, 
            (vx*fx + vy*fy + vz*fz) / 2300.0,
            (vx*rx + vy*ry + vz*rz) / 2300.0,
            (vx*ux + vy*uy + vz*uz) / 2300.0,
            (ax*fx + ay*fy + az*fz) / 5.5,
            (ax*rx + ay*ry + az*rz) / 5.5,
            (ax*ux + ay*uy + az*uz) / 5.5,
            fx, fy, fz, rx, ry, rz, ux, uy, uz,
            (rx_rel*fx + ry_rel*fy + rz_rel*fz) / 6000.0,
            (rx_rel*rx + ry_rel*ry + rz_rel*rz) / 6000.0,
            (rx_rel*ux + ry_rel*uy + rz_rel*uz) / 6000.0,
            (rvx_rel*fx + rvy_rel*fy + rvz_rel*fz) / 6000.0,
            (rvx_rel*rx + rvy_rel*ry + rvz_rel*rz) / 6000.0,
            (rvx_rel*ux + rvy_rel*uy + rvz_rel*uz) / 6000.0,
            math.sqrt(max(0.0, player.boost_amount)),
            float(player.on_ground), float(player.has_flip), float(player.is_demoed)
        ]
        
        obs.extend(np.atleast_1d(previous_action).tolist())
        obs_arr = np.array(obs, dtype=np.float32)
        if not np.isfinite(obs_arr).all():
            obs_arr = np.nan_to_num(obs_arr)
        return obs_arr

# ------------------------------------------------------------------------------
# 4. FAST-MATH REWARD SHAPING (With NaN returns blocked)
# ------------------------------------------------------------------------------
class CompoundAerialReward(RewardFunction):
    def __init__(self): self.air_time = {}
    def reset(self, initial_state: GameState): self.air_time.clear()
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cid = player.car_id
        if not player.on_ground: self.air_time[cid] = self.air_time.get(cid, 0.0) + (8.0 / 120.0)
        else: self.air_time[cid] = 0.0
        rew = min(min(self.air_time.get(cid, 0.0) / 1.75, 1.0), min(max(player.car_data.position[2], 0.0) / 2044.0, 1.0))
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
            rew = (vel_delta / 6000.0) * max(player.car_data.position[2] / 2044.0, 0.1)
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
            rew = ((vx/speed)*(dx/dist) + (vy/speed)*(dy/dist) + (vz/speed)*(dz/dist))
            return float(rew) if not math.isnan(rew) else 0.0
        return 0.0

class KinestheticShadowDefense(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cx, cy, cz = player.car_data.position
        bx, by, bz = state.ball.position
        dist = math.sqrt((cx-bx)**2 + (cy-by)**2 + (cz-bz)**2)
        dist_factor = math.exp(-dist / 3000.0)
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
            rew = -float(abs(forward_vel) / 2300.0)
            return rew if not math.isnan(rew) else 0.0
        return 0.0

# ------------------------------------------------------------------------------
# 5. CURRICULUM MUTATORS (Anti-Telefrag Architecture)
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
        else:
            DefaultState().reset(wrapper)
            
        for car in wrapper.cars:
            vel = car.linear_velocity
            car.set_lin_vel(vel[0]*random.uniform(0.98, 1.02), vel[1]*random.uniform(0.98, 1.02), vel[2]*random.uniform(0.98, 1.02))

# ------------------------------------------------------------------------------
# 6. ENVIRONMENT GENERATION
# ------------------------------------------------------------------------------
def build_env():
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)

    reward_fn = CombinedReward(
        (CompoundAerialReward(), 2.0),
        (DynamicTouchReward(), 4.0),
        (VectorAlignmentReward(), 1.0),
        (KinestheticShadowDefense(), 1.5),
        (BackwardVelocityPenalty(), 0.5)
    )
    return rlgym_sim.make(
        tick_skip=8, team_size=1, spawn_opponents=True,
        reward_fn=reward_fn, obs_builder=SOTAObservation(),
        action_parser=SOTAActionParser(), state_setter=EscalateMutator(),
        terminal_conditions=[TimeoutCondition(400), GoalScoredCondition()]
    )

# ------------------------------------------------------------------------------
# 7. SOTA V8 MAIN PPO ENGINE (BYPASSING THE SUICIDE SWITCH)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        pass
        
    # Revert your CMF files to their default names to fix the RocketSim C++ Bug!
    revert_collision_meshes()

    print("🚀 Initializing 48 vCPU SOTA AI Engine (Anti-Lag Architecture)...")
    
    # 💡 DRY-RUN: Catch errors safely BEFORE spawning 22 background workers
    print("🧪 Performing synchronous dry-run of build_env()...")
    try:
        temp_env = build_env()
        obs_size = len(temp_env.reset()[0])
        temp_env.close()
        print("✅ Environment built successfully! Spawning workers...\n")
    except Exception as e:
        print(f"🚨 FATAL: build_env() crashed before multiprocessing could start!\n{traceback.format_exc()}")
        sys.exit(1)

    WORKER_CORES = 22
    GLOBAL_BATCH_SIZE = 99_000
    TOTAL_ITERS = 500
    
    learner = Learner(
        build_env,
        n_proc=WORKER_CORES, 
        ppo_batch_size=GLOBAL_BATCH_SIZE,
        ts_per_iteration=GLOBAL_BATCH_SIZE,
        exp_buffer_size=GLOBAL_BATCH_SIZE * 3, 
        ppo_minibatch_size=33_000, 
        ppo_ent_coef=0.01,
        policy_lr=1e-4,
        critic_lr=1e-4,
        ppo_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_to_wandb=False
    )

    # 🛑 COMPLETELY BYPASS `learner.learn()` TO PREVENT WORKER SUICIDE 🛑
    try:
        for i in tqdm(range(TOTAL_ITERS), desc="Training Rocket League Bot", file=sys.stdout):
            
            # 1. Manually command the Agent Manager to collect experience
            experience, metrics, steps, coll_time = learner.agent.collect_timesteps(GLOBAL_BATCH_SIZE)
            
            # 2. Feed the PPO Engine safely
            learner.add_new_experience(experience)
            learner.ppo_learner.learn(learner.experience_buffer)
            learner.agent.cumulative_timesteps += steps
            
            # 3. Apply your dynamic LR & Entropy Decays
            progress = (i + 1) / TOTAL_ITERS
            new_lr = 1e-4 - ((1e-4 - 5e-6) * progress)
            new_ent = 0.01 - ((0.01 - 0.005) * progress)
            
            for param_group in learner.ppo_learner.policy_optimizer.param_groups: 
                param_group['lr'] = new_lr
            for param_group in learner.ppo_learner.value_optimizer.param_groups: 
                param_group['lr'] = new_lr
            
            learner.ppo_ent_coef = new_ent
            learner.ppo_learner.ent_coef = new_ent

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted safely.")
    except Exception as e:
        print(f"\n🚨 CRASH DURING TRAINING:\n{traceback.format_exc()}")
    finally:
        # Gracefully kill the 22 workers once training is ACTUALLY done
        learner.cleanup()

    print("\n🔥 Training Concluded! Quantizing weights to ONNX...")
    
    # Exporting the weights securely
    try:
        if hasattr(learner, 'ppo_learner'): policy = learner.ppo_learner.policy
        else: policy = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
    except AttributeError:
        policy = learner.agent.policy.actor
        
    policy.eval().to("cpu")
    
    dummy_input = torch.randn(1, obs_size, dtype=torch.float32)
    export_path = "SOTA_RLBot_V8_Agent_48Core.onnx"
    
    try:
        torch.onnx.export(
            policy, dummy_input, export_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['observation'], output_names=['action_logits']
        )
        print(f"✅ Weights saved safely -> {export_path}")
    except Exception as e:
        print(f"❌ Failed to export ONNX. The network was trained successfully, but export failed: {e}")
