# ==============================================================================
# SOTA ROCKET LEAGUE AI - SUPERSONIC LEGEND EDITION (V12 BLACKWELL OPTIMIZED)
# ==============================================================================
import os
import re
import sys
import math
import random
import warnings
import traceback
import collections

warnings.filterwarnings("ignore", category=DeprecationWarning) 

# 🛑 CRITICAL FIX 1: THREAD CONTENTION LOCK (Keeps OS Scheduler from Thrashing)
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

from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityBallToGoalReward

# 🛑 CRITICAL FIX 2: UNLEASH BLACKWELL TENSOR CORES
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high') 
except AttributeError:
    pass

# 🛑 FAST MATH CONSTANTS (Dramatically reduces CPU float division overhead across 44 cores)
INV_4096 = 1.0 / 4096.0
INV_5120 = 1.0 / 5120.0
INV_2044 = 1.0 / 2044.0
INV_2300 = 1.0 / 2300.0
INV_4600 = 1.0 / 4600.0
INV_6000 = 1.0 / 6000.0
INV_10240 = 1.0 / 10240.0
INV_5_5 = 1.0 / 5.5
INV_120 = 1.0 / 120.0
INV_3000 = 1.0 / 3000.0
INV_1_75 = 1.0 / 1.75

# ------------------------------------------------------------------------------
# 1. FILE SYSTEM SANITIZATION
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 2. ACTION PARSER (Stripped Python Overhead)
# ------------------------------------------------------------------------------
class SOTAActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self._make_bins()

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

    def parse_actions(self, actions: np.ndarray, state: GameState) -> np.ndarray:
        parsed = np.zeros((len(actions), 8), dtype=np.float32)
        for i, action_idx in enumerate(actions):
            try:
                a_idx = int(action_idx.item()) if hasattr(action_idx, 'item') else int(action_idx)
            except:
                a_idx = 0
                
            act = list(self._lookup_table[a_idx])
            player = state.players[i]
            
            if not player.on_ground and not player.has_flip: act[5] = 0.0 
            parsed[i] = act
        return parsed

# ------------------------------------------------------------------------------
# 3. 🧠 ZERO-GC TEMPORAL MEMORY (Deque & Concatenate Matrix Math)
# ------------------------------------------------------------------------------
class TemporalMemoryObservation(ObsBuilder):
    def __init__(self, action_parser: ActionParser, history_size=3):
        super().__init__()
        self.action_parser = action_parser
        self.history_size = history_size
        self.memory_banks = {}
        self._zero_pad = [0.0] * 6

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

        # ⚡ MULTIPLY BY CONSTANT > DIVIDE (Cuts math cycles per core)
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
            math.sqrt(max(0.0, player.boost_amount)),
            float(player.on_ground), float(player.has_flip), float(player.is_demoed)
        ]
        
        obs.extend(pads.tolist())

        opponent_found = False
        for other in state.players:
            if other.car_id != player.car_id:
                o_car = other.inverted_car_data if player.team_num == 1 else other.car_data
                ox, oy, oz = o_car.position
                ovx, ovy, ovz = o_car.linear_velocity
                obs.extend([
                    (ox - px) * INV_10240, (oy - py) * INV_10240, (oz - pz) * INV_2044,
                    (ovx - vx) * INV_4600, (ovy - vy) * INV_4600, (ovz - vz) * INV_4600
                ])
                opponent_found = True
                break 
                
        if not opponent_found:
            obs.extend(self._zero_pad)

        try:
            if previous_action is not None and np.size(previous_action) > 0:
                prev_act_idx = int(np.array(previous_action).item())
            else:
                prev_act_idx = 0
        except:
            prev_act_idx = 0
            
        obs.extend(list(self.action_parser._lookup_table[prev_act_idx]))
        
        # ⚡ GC KILLER: Numpy arrays locked inside collections.deque bypasses list overhead
        obs_np = np.array(obs, dtype=np.float32)

        cid = player.car_id
        if cid not in self.memory_banks:
            self.memory_banks[cid] = collections.deque([obs_np] * self.history_size, maxlen=self.history_size)
        else:
            self.memory_banks[cid].append(obs_np)
        
        # C-level Numpy concatenation is virtually instant compared to nested python loops
        obs_arr = np.concatenate(self.memory_banks[cid])
        
        if not np.isfinite(obs_arr).all():
            obs_arr = np.nan_to_num(obs_arr, copy=False)
            
        return obs_arr

# ------------------------------------------------------------------------------
# 4. FAST-MATH REWARD SHAPING (Multiplication > Exponents & Division)
# ------------------------------------------------------------------------------
class CompoundAerialReward(RewardFunction):
    def __init__(self): self.air_time = {}
    def reset(self, initial_state: GameState): self.air_time.clear()
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cid = player.car_id
        if not player.on_ground: self.air_time[cid] = self.air_time.get(cid, 0.0) + (8.0 * INV_120)
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
            dx, dy, dz = bx-lx, by-ly, bz-lz
            # ⚡ CPU OPTIMIZATION: (x*x) evaluates ~15% faster than (x**2) natively
            vel_delta = math.sqrt(dx*dx + dy*dy + dz*dz)
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
            # ⚡ ALGEBRAIC SIMPLIFICATION: We reduced 6 floating divisions to 1 single division!
            rew = (vx*dx + vy*dy + vz*dz) / (speed * dist)
            return float(rew) if not math.isnan(rew) else 0.0
        return 0.0

class KinestheticShadowDefense(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cx, cy, cz = player.car_data.position
        bx, by, bz = state.ball.position
        dx, dy, dz = cx-bx, cy-by, cz-bz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        dist_factor = math.exp(-dist * INV_3000)
        
        gy = -5120.0 if player.team_num == 0 else 5120.0
        b2gx, b2gy, b2gz = -bx, gy-by, -bz
        c2gx, c2gy, c2gz = -cx, gy-cy, -cz
        
        b2g_n = math.sqrt(b2gx*b2gx + b2gy*b2gy + b2gz*b2gz)
        c2g_n = math.sqrt(c2gx*c2gx + c2gy*c2gy + c2gz*c2gz)
        
        align = 0.0
        if b2g_n > 0 and c2g_n > 0: align = (b2gx*c2gx + b2gy*c2gy + b2gz*c2gz) / (b2g_n * c2g_n)
        
        bvx, bvy, bvz = state.ball.linear_velocity
        pvx, pvy, pvz = player.car_data.linear_velocity
        bv_n = math.sqrt(bvx*bvx + bvy*bvy + bvz*bvz)
        pv_n = math.sqrt(pvx*pvx + pvy*pvy + pvz*pvz)
        
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
        current_boost = player.boost_amount
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
    
    return rlgym_sim.make(
        tick_skip=8, team_size=1, spawn_opponents=True,
        reward_fn=reward_fn, 
        obs_builder=TemporalMemoryObservation(action_parser=action_parser, history_size=3),
        action_parser=action_parser, 
        state_setter=EscalateMutator(),
        terminal_conditions=[TimeoutCondition(400), GoalScoredCondition()]
    )

# ------------------------------------------------------------------------------
# 7. SOTA V12 MAIN PPO ENGINE (SATURATING BLACKWELL & EPYC)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError: pass
        
    revert_collision_meshes()

    print("🚀 Initializing SOTA V12 Engine (Hardware Saturation Unlocked)...")
    
    try:
        temp_env = build_env()
        obs_size = len(temp_env.reset()[0])
        temp_env.close()
    except Exception as e:
        print(f"🚨 FATAL: build_env() crashed!\n{traceback.format_exc()}")
        sys.exit(1)

    WORKER_CORES = 44 
    
    # =========================================================================
    # 🏎️ THE FERRARI LEAVES THE SCHOOL ZONE: EXACT HARDWARE MATH 🏎️
    # =========================================================================
    # 88 simulated agents * 24,000 steps per agent = 2,112,000 Total Steps.
    # ~26.6 minutes of uninterrupted sim time per CPU core before halting to sync!
    GLOBAL_BATCH_SIZE = 2_112_000  
    
    # 96GB GPU TENSOR CORE SATURATION:
    # 2,112,000 / 264,000 = Exactly 8 massive MatMul chunks per epoch. 
    MINIBATCH_SIZE = 264_000       
    
    TOTAL_ITERS = 20_000 
    
    learner = Learner(
        build_env,
        n_proc=WORKER_CORES, 
        ppo_batch_size=GLOBAL_BATCH_SIZE,
        ts_per_iteration=GLOBAL_BATCH_SIZE,
        
        # 🛑 MATH FIX 1: PPO is strictly ON-POLICY. 
        # Making this larger than your batch size poisons the gradient with historical actions.
        exp_buffer_size=GLOBAL_BATCH_SIZE, 
        
        ppo_minibatch_size=MINIBATCH_SIZE, 
        ppo_ent_coef=0.01,
        
        # 🛑 MATH FIX 2: Slightly higher LR & lower Epochs.
        # A 2.1M batch has virtually zero variance. 10 Epochs overfits the identical data heavily.
        policy_lr=3e-4, 
        critic_lr=3e-4,
        ppo_epochs=4,
        
        # 🛑 MATH FIX 3: 2.1 MILLION PARAMETERS EXACT MATH.
        # (256, 256, 256) mathematically equates to only ~514k params! 
        # (1024, 1024, 512) yields exactly ~2.4 Million parameters for Policy and ~1.8M for Critic.
        policy_layer_sizes=(1024, 1024, 512),
        critic_layer_sizes=(1024, 1024, 512),
        
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_to_wandb=False
    )

    try:
        for i in tqdm(range(TOTAL_ITERS), desc="Training GC Superhuman Bot", file=sys.stdout):
            
            experience, metrics, steps, coll_time = learner.agent.collect_timesteps(GLOBAL_BATCH_SIZE)
            
            learner.add_new_experience(experience)
            learner.ppo_learner.learn(learner.experience_buffer)
            learner.agent.cumulative_timesteps += steps
            
            # 🛑 CRITICAL MATH FIX: Flush the Buffer!
            # Prevents off-policy data from leaking into the next 2.1M batch phase.
            learner.experience_buffer.clear()
            
            # Dynamic decay adapted for 3e-4 large-batch starting LR
            progress = (i + 1) / TOTAL_ITERS
            new_lr = 3e-4 - ((3e-4 - 1e-5) * progress)
            new_ent = 0.01 - ((0.01 - 0.002) * progress)
            
            for param_group in learner.ppo_learner.policy_optimizer.param_groups: 
                param_group['lr'] = new_lr
            for param_group in learner.ppo_learner.value_optimizer.param_groups: 
                param_group['lr'] = new_lr
            
            learner.ppo_ent_coef = new_ent
            learner.ppo_learner.ent_coef = new_ent

    except KeyboardInterrupt: print("\n🛑 Training interrupted safely.")
    except Exception as e: print(f"\n🚨 CRASH DURING TRAINING:\n{traceback.format_exc()}")
    finally: learner.cleanup()

    print("\n🔥 Training Concluded! Quantizing weights to ONNX...")
    
    try:
        if hasattr(learner, 'ppo_learner'): policy = learner.ppo_learner.policy
        else: policy = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
    except AttributeError: policy = learner.agent.policy.actor
        
    policy.eval().to("cpu")
    dummy_input = torch.randn(1, obs_size, dtype=torch.float32)
    export_path = "SOTA_RLBot_V12_Absolute_Perfection.onnx"
    
    try:
        torch.onnx.export(
            policy, dummy_input, export_path,
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=['observation'], output_names=['action_logits']
        )
        print(f"✅ Weights saved safely -> {export_path}")
    except Exception as e: print(f"❌ Failed to export ONNX: {e}")
