# ==============================================================================
# SOTA ROCKET LEAGUE AI - SIM-TO-REAL IMMORTAL ENGINE (SOTA V120)
# 40-Core EPYC / 1v1 ONLY / 1000-Point Goals / 5k Iters / Zero Useless Actions
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
import multiprocessing as mp

# 🛑 SILENCE ANNOYING PYTORCH & PYTHON WARNINGS 🛑
warnings.filterwarnings("ignore")
logging.getLogger("torch.onnx").setLevel(logging.ERROR)
logging.getLogger("torch.export").setLevel(logging.ERROR)

# 🛑 CRITICAL FIX 1: KILL "THREAD BOMB" DURING ROLLOUTS 🛑
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

from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, NoTouchTimeoutCondition
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityBallToGoalReward, VelocityPlayerToBallReward

# Worker processes strictly locked to 1 thread for fast simulation
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
INV_8300 = 1.0 / 8300.0
INV_10240= 1.0 / 10240.0
INV_3000 = 1.0 / 3000.0

# Hitbox normalization constants
INV_150 = 1.0 / 150.0  
INV_100 = 1.0 / 100.0  
INV_50  = 1.0 / 50.0   

# ------------------------------------------------------------------------------
# 1. DOMAIN RANDOMIZATION WRAPPERS (Sim-to-Real Protection)
# ------------------------------------------------------------------------------
class ActionDelayWrapper(gym.Wrapper):
    def __init__(self, env, action_parser, min_delay=0, max_delay=1):
        super().__init__(env)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.action_buffer = []
        self.current_delay = 0
        self.idle_action_idx = action_parser.get_idle_action_idx()

    def reset(self, **kwargs):
        self.action_buffer.clear()
        self.current_delay = random.randint(self.min_delay, self.max_delay) 
        return self.env.reset(**kwargs)

    def step(self, action):
        action_arr = np.array(action, copy=True)
        
        # Dynamically matches exact array shape safely (flawless for 1v1s)
        if len(self.action_buffer) == 0 and self.current_delay > 0:
            idle_arr = np.full_like(action_arr, self.idle_action_idx)
            for _ in range(self.current_delay):
                self.action_buffer.append(idle_arr)

        self.action_buffer.append(action_arr)
        
        if len(self.action_buffer) > self.current_delay:
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
                vel[0] * random.uniform(0.98, 1.02) + random.uniform(-10.0, 10.0),
                vel[1] * random.uniform(0.98, 1.02) + random.uniform(-10.0, 10.0),
                vel[2] * random.uniform(0.98, 1.02) + random.uniform(-10.0, 10.0)
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
# 2. VECTORIZED ACTION PARSER (Masterclass 648-Bin Compression)
# ------------------------------------------------------------------------------
class SOTAActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = np.array(self._make_bins(), dtype=np.float32)

    def _make_bins(self):
        bins = []
        for throttle in [-1.0, 0.0, 1.0]:
            # 🏆 GENIUS CONFIG: Steer & Yaw locked to shared variable. 
            # Reduces 1,944 combinations to 648 seamlessly.
            for steer_yaw in [-1.0, 0.0, 1.0]:
                for pitch in [-1.0, 0.0, 1.0]:
                    for roll in [-1.0, 0.0, 1.0]:
                        for jump in [0.0, 1.0]:
                            for boost in [0.0, 1.0]:
                                for handbrake in [0.0, 1.0]:
                                    bins.append([throttle, steer_yaw, pitch, steer_yaw, roll, jump, boost, handbrake])
        return bins
        
    def get_idle_action_idx(self):
        for i, b in enumerate(self._lookup_table):
            if np.all(b == 0.0):
                return i
        return 0

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.int32).flatten()
        actions = np.clip(actions, 0, len(self._lookup_table) - 1)
        parsed = self._lookup_table[actions].copy()
        return parsed

# ------------------------------------------------------------------------------
# 3. ULTRA-FAST OBSERVATION BUILDER (The Ghost Padding Matrix)
# ------------------------------------------------------------------------------
class TemporalMemoryObservation(ObsBuilder):
    def __init__(self, action_parser: ActionParser, history_size=1):
        super().__init__()
        self.action_parser = action_parser
        # 🛑 1V1 PRESERVATION TRICK: 
        # Leaving these at 3 and 2 guarantees the NN input stays EXACTLY at 156 variables.
        # This allows 100% flawless auto-resuming from 3v3 models into 1v1 without crashing!
        self.MAX_OPPONENTS = 3
        self.MAX_TEAMMATES = 2

    def reset(self, initial_state: GameState): 
        pass 

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
        rx, ry, rz = car.right() 

        h_len = player.car_data.hitbox_size[0] if hasattr(player.car_data, 'hitbox_size') else 118.01 
        h_wid = player.car_data.hitbox_size[1] if hasattr(player.car_data, 'hitbox_size') else 84.20
        h_hei = player.car_data.hitbox_size[2] if hasattr(player.car_data, 'hitbox_size') else 36.16

        dx, dy, dz = bx - px, by - py, bz - pz
        dvx, dvy, dvz = bvx - vx, bvy - vy, bvz - vz
        
        local_bx = dx*fx + dy*fy + dz*fz
        local_by = dx*rx + dy*ry + dz*rz
        local_bz = dx*ux + dy*uy + dz*uz
        
        local_bvx = dvx*fx + dvy*fy + dvz*fz
        local_bvy = dvx*rx + dvy*ry + dvz*rz
        local_bvz = dvx*ux + dvy*uy + dvz*uz

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
            
            local_bx * INV_10240, local_by * INV_10240, local_bz * INV_10240, 
            local_bvx * INV_8300, local_bvy * INV_8300, local_bvz * INV_8300, 
            
            math.sqrt(max(0.0, player.boost_amount)),
            
            float(player.on_ground), float(player.has_flip), float(player.is_demoed),
            
            h_len * INV_150, h_wid * INV_100, h_hei * INV_50
        ]

        obs.extend(pads.tolist())

        true_bx, true_by, true_bz = state.ball.position

        opponents = [other for other in state.players if other.team_num != player.team_num]
        opponents.sort(key=lambda x: (x.car_data.position[0]-true_bx)**2 + 
                                     (x.car_data.position[1]-true_by)**2 + 
                                     (x.car_data.position[2]-true_bz)**2)
        
        added_opps = 0
        for other in opponents:
            if added_opps >= self.MAX_OPPONENTS:
                break
                
            o_car = other.inverted_car_data if player.team_num == 1 else other.car_data
            ox, oy, oz = o_car.position
            ovx, ovy, ovz = o_car.linear_velocity
            
            ofx, ofy, ofz = o_car.forward()
            orx, o_ry, orz = o_car.right()
            oux, ouy, ouz = o_car.up()
            
            odx, ody, odz = ox - px, oy - py, oz - pz
            odvx, odvy, odvz = ovx - vx, ovy - vy, ovz - vz
            
            local_ox = odx*fx + ody*fy + odz*fz
            local_oy = odx*rx + ody*ry + odz*rz
            local_oz = odx*ux + ody*uy + odz*uz
            
            local_ovx = odvx*fx + odvy*fy + odvz*fz
            local_ovy = odvx*rx + odvy*ry + odvz*rz
            local_ovz = odvx*ux + odvy*uy + odvz*uz

            obs.extend([
                local_ox * INV_10240, local_oy * INV_10240, local_oz * INV_10240,
                local_ovx * INV_4600, local_ovy * INV_4600, local_ovz * INV_4600,
                ofx, ofy, ofz, orx, o_ry, orz, oux, ouy, ouz,
                math.sqrt(max(0.0, other.boost_amount))
            ])
            added_opps += 1
            
        for _ in range(self.MAX_OPPONENTS - added_opps):
            # In 1v1, the missing 3v3 opponents safely get padded out of bounds here!
            obs.extend([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        teammates = [other for other in state.players if other.team_num == player.team_num and other.car_id != player.car_id]
        teammates.sort(key=lambda x: (x.car_data.position[0]-true_bx)**2 + 
                                     (x.car_data.position[1]-true_by)**2 + 
                                     (x.car_data.position[2]-true_bz)**2)
        
        added_tm8s = 0
        for other in teammates:
            if added_tm8s >= self.MAX_TEAMMATES:
                break
                
            t_car = other.inverted_car_data if player.team_num == 1 else other.car_data
            tx, ty, tz = t_car.position
            tvx, tvy, tvz = t_car.linear_velocity
            
            tfx, tfy, tfz = t_car.forward()
            trx, t_ry, trz = t_car.right()
            tux, tuy, tuz = t_car.up()
            
            tdx, tdy, tdz = tx - px, ty - py, tz - pz
            tdvx, tdvy, tdvz = tvx - vx, tvy - vy, tvz - vz
            
            local_tx = tdx*fx + tdy*fy + tdz*fz
            local_ty = tdx*rx + tdy*ry + tdz*rz
            local_tz = tdx*ux + tdy*uy + tdz*uz
            
            local_tvx = tdvx*fx + tdvy*fy + tdvz*fz
            local_tvy = tdvx*rx + tdvy*ry + tdvz*rz
            local_tvz = tdvx*ux + tdvy*uy + tdvz*uz

            obs.extend([
                local_tx * INV_10240, local_ty * INV_10240, local_tz * INV_10240,
                local_tvx * INV_4600, local_tvy * INV_4600, local_tvz * INV_4600,
                tfx, tfy, tfz, trx, t_ry, trz, tux, tuy, tuz,
                math.sqrt(max(0.0, other.boost_amount))
            ])
            added_tm8s += 1
            
        for _ in range(self.MAX_TEAMMATES - added_tm8s):
            # In 1v1, the missing 3v3 teammates safely get padded out of bounds here!
            obs.extend([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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

        obs_arr = np.array(obs, dtype=np.float32)
        if not np.isfinite(obs_arr).all():
            obs_arr = np.nan_to_num(obs_arr)
            
        return obs_arr

# ------------------------------------------------------------------------------
# 4. ALGEBRAICALLY PERFECT REWARD SHAPING (The 1v1 Striker Math)
# ------------------------------------------------------------------------------
class OffensivePushReward(RewardFunction):
    """
    The Striker's Breadcrumb.
    Rewards the bot for driving fast at the ball ONLY if it is on the correct side 
    of the play (pushing the ball towards the enemy net). Cures "Freestyler Syndrome."
    """
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        bx, by, bz = state.ball.position
        px, py, pz = player.car_data.position
        
        gy = 5120.0 if player.team_num == 0 else -5120.0
        
        c2bx, c2by, c2bz = bx - px, by - py, bz - pz
        c2b_mag = math.sqrt(c2bx**2 + c2by**2 + c2bz**2)
        
        b2gx, b2gy, b2gz = 0.0 - bx, gy - by, 0.0 - bz
        b2g_mag = math.sqrt(b2gx**2 + b2gy**2 + b2gz**2)
        
        if c2b_mag > 0 and b2g_mag > 0:
            alignment = (c2bx*b2gx + c2by*b2gy + c2bz*b2gz) / (c2b_mag * b2g_mag)
            
            # Ensures the bot approaches from BEHIND the ball relative to the enemy net
            if alignment > 0:
                vx, vy, vz = player.car_data.linear_velocity
                vel_to_ball = (vx*c2bx + vy*c2by + vz*c2bz) / c2b_mag
                if vel_to_ball > 0:
                    return float(alignment * (vel_to_ball * INV_2300))
        return 0.0

class CompoundAerialReward(RewardFunction):
    def reset(self, initial_state: GameState): pass
    
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        px, py, pz = player.car_data.position
        
        if player.on_ground or pz < 250.0 or state.ball.position[2] < 350.0:
            return 0.0 
        
        bx, by, bz = state.ball.position
        bvx, bvy, bvz = state.ball.linear_velocity
        vx, vy, vz = player.car_data.linear_velocity
        
        pred_bx = bx + (bvx * 0.4)
        pred_by = by + (bvy * 0.4)
        pred_bz = bz + (bvz * 0.4) - 52.0 
        
        dx, dy, dz = pred_bx - px, pred_by - py, pred_bz - pz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        shaping_rew = 0.0
        if dist > 0:
            vel_to_pred_ball = (vx*dx + vy*dy + vz*dz) / dist
            height_mult = max(0.0, (pz - 250.0) * INV_2044)
            shaping_rew = max(0.0, vel_to_pred_ball * INV_2300) * height_mult * 2.0 
            
        touch_rew = 0.0
        if player.ball_touched:
            height_frac = min(max(pz, 0.0) * INV_2044, 1.0) 
            touch_rew = float(height_frac) * 20.0 
            
        return float(shaping_rew + touch_rew)

class KinestheticShadowDefense(RewardFunction):
    def reset(self, initial_state: GameState): pass
    def get_reward(self, player: PlayerData, state: GameState, prev_action: np.ndarray) -> float:
        cx, cy, cz = player.car_data.position
        bx, by, bz = state.ball.position
        
        gy = -5120.0 if player.team_num == 0 else 5120.0
        b2gx, b2gy, b2gz = -bx, gy-by, -bz
        c2gx, c2gy, c2gz = -cx, gy-cy, -cz
        
        b2g_n = math.sqrt(b2gx**2 + b2gy**2 + b2gz**2)
        c2g_n = math.sqrt(c2gx**2 + c2gy**2 + c2gz**2)
        
        out_of_position = c2g_n > b2g_n

        if out_of_position:
            vx, vy, vz = player.car_data.linear_velocity
            if c2g_n > 0:
                vel_to_goal = (vx*c2gx + vy*c2gy + vz*c2gz) / c2g_n
                return float(max(0.0, vel_to_goal * INV_2300) * 0.1) 
            return 0.0

        dist = math.sqrt((cx-bx)**2 + (cy-by)**2 + (cz-bz)**2)
        dist_factor = math.exp(-dist * INV_3000)
        
        align = 0.0
        if b2g_n > 0 and c2g_n > 0: 
            align = (b2gx*c2gx + b2gy*c2gy + b2gz*c2gz) / (b2g_n * c2g_n)

        bvx, bvy, bvz = state.ball.linear_velocity
        pvx, pvy, pvz = player.car_data.linear_velocity
        bv_n = math.sqrt(bvx**2 + bvy**2 + bvz**2)
        pv_n = math.sqrt(pvx**2 + pvy**2 + pvz**2)
        
        v_match = 0.0
        if bv_n > 0 and pv_n > 0: 
            v_match = (bvx*pvx + bvy*pvy + bvz*pvz) / (bv_n * pv_n)
        
        v_mult = 1.0 + (v_match * 0.5) 
        
        return float(dist_factor * max(0.0, align) * v_mult)

# ------------------------------------------------------------------------------
# 5. CURRICULUM MUTATORS (The 1v1 Gauntlet)
# ------------------------------------------------------------------------------
class EscalateMutator(StateSetter):
    def reset(self, wrapper: StateWrapper):
        scenario = random.random()
        
        if scenario < 0.30:
            # 🎯 30% KICKOFFS: The most important mechanic in 1v1.
            DefaultState().reset(wrapper)
                
        elif scenario < 0.50:
            # 🎯 20% THE STRIKER PACK (Direct Shooting Practice)
            team_side = random.choice([-1.0, 1.0])
            wrapper.ball.set_pos(random.uniform(-1000, 1000), 2000.0 * team_side, 100.0)
            wrapper.ball.set_lin_vel(0.0, 1000.0 * team_side, 0.0)
            
            for car in wrapper.cars:
                if (team_side == 1.0 and car.team_num == 0) or (team_side == -1.0 and car.team_num == 1):
                    # Attacker placed directly behind ball
                    car.set_pos(wrapper.ball.position[0] + random.uniform(-200, 200), 1000.0 * team_side, 17.05)
                    car.set_rot(0.0, (math.pi/2) * team_side, 0.0)
                    car.set_lin_vel(0.0, 1500.0 * team_side, 0.0)
                    car.boost = random.uniform(0.5, 1.0)
                else:
                    # Defender in goal
                    car.set_pos(random.uniform(-800, 800), 5100.0 * team_side, 17.05)
                    car.set_rot(0.0, (math.pi/2) * -team_side, 0.0)
                    car.set_lin_vel(0.0, 0.0, 0.0)
                    car.boost = random.uniform(0.1, 0.5)

        elif scenario < 0.70:
            # 🛡️ 20% SIDE WALL / CROSS DEFENSE
            side = random.choice([-1.0, 1.0])
            y_dir = random.choice([-1.0, 1.0]) 
            for car in wrapper.cars:
                is_defending = (y_dir == 1.0 and car.team_num == 1) or (y_dir == -1.0 and car.team_num == 0)
                if is_defending:
                    car.set_pos(random.uniform(-800, 800), 5100.0 * y_dir, 17.05)
                    car.set_rot(0.0, (math.pi/2) * -y_dir, 0.0) 
                    car.set_lin_vel(0.0, 0.0, 0.0)
                else:
                    car.set_pos(3000.0 * side + random.uniform(-500, 500), -500.0 * y_dir + random.uniform(-500, 500), 200.0)
                    car.set_rot(0.0, (math.pi/2) * y_dir, 0.0) 
                    car.set_lin_vel(0.0, 1500.0 * y_dir, 600.0)
                car.boost = random.uniform(0.1, 1.0)
            
            wrapper.ball.set_pos(3000.0 * side, 100.0 * y_dir, 900.0)
            wrapper.ball.set_lin_vel(0.0, 1500.0 * y_dir, 600.0)
            
        elif scenario < 0.85:
            # ✈️ 15% AERIAL INTERCEPTS
            wrapper.ball.set_pos(random.uniform(-2000, 2000), random.uniform(-2000, 2000), random.uniform(800, 1500))
            wrapper.ball.set_lin_vel(random.uniform(-800, 800), random.uniform(-800, 800), random.uniform(300, 700))
            for car in wrapper.cars:
                car.set_pos(random.uniform(-3000, 3000), random.uniform(-4000, 4000), random.uniform(500, 1500))
                car.set_rot(random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))
                car.set_ang_vel(random.uniform(-5, 5), random.uniform(-5, 5), random.uniform(-5, 5))
                car.boost = random.uniform(0.0, 0.5)

        else:
            # 🎮 15% GROUND DRIBBLES & FLICKS (The 1v1 special)
            team_side = random.choice([-1.0, 1.0])
            wrapper.ball.set_pos(random.uniform(-1000, 1000), random.uniform(-2000, 2000), 200.0)
            wrapper.ball.set_lin_vel(0.0, random.uniform(500, 1000) * team_side, 0.0)
            for car in wrapper.cars:
                if (team_side == 1.0 and car.team_num == 0) or (team_side == -1.0 and car.team_num == 1):
                    # Attacker perfectly underneath ball
                    car.set_pos(wrapper.ball.position[0], wrapper.ball.position[1] - (100 * team_side), 17.05)
                    car.set_rot(0.0, (math.pi/2) * team_side, 0.0)
                    car.set_lin_vel(0.0, wrapper.ball.linear_velocity[1], 0.0)
                    car.boost = 1.0
                else:
                    # Defender waiting in net
                    car.set_pos(random.uniform(-800, 800), 5100.0 * team_side, 17.05)
                    car.set_rot(0.0, (math.pi/2) * -team_side, 0.0)
                    car.set_lin_vel(0.0, 0.0, 0.0)
                    car.boost = random.uniform(0.1, 0.5)

# ------------------------------------------------------------------------------
# 6. ENVIRONMENT GENERATION
# ------------------------------------------------------------------------------
def build_env():
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)

    # 🛑 V120 1v1 ANTI-USELESS MATRIX:
    # 1000 Point Goal. Demolitions heavily rewarded for 1v1 control!
    reward_fn = CombinedReward(
        (
            EventReward(goal=1000.0, concede=-1000.0, shot=50.0, save=50.0, demo=15.0), 
            VelocityBallToGoalReward(),            
            OffensivePushReward(), 
            VelocityPlayerToBallReward(), 
            CompoundAerialReward(),       
            KinestheticShadowDefense()
        ),
        (1.0, 0.15, 0.08, 0.02, 0.05, 0.02)    
    )
    
    action_parser = SOTAActionParser()
    robust_state_setter = PhysicsRandomizationMutator(EscalateMutator())
    
    # 🛑 1V1 MODE ACTIVATED: Team size set strictly to 1.
    env = rlgym_sim.make(
        tick_skip=8, team_size=1, spawn_opponents=True,
        reward_fn=reward_fn, 
        obs_builder=TemporalMemoryObservation(action_parser=action_parser, history_size=1),
        action_parser=action_parser, 
        state_setter=robust_state_setter,
        terminal_conditions=[TimeoutCondition(1500), GoalScoredCondition(), NoTouchTimeoutCondition(225)]
    )
    
    env = ActionDelayWrapper(env, action_parser, min_delay=0, max_delay=1)
    return env

# ------------------------------------------------------------------------------
# 7. SOTA V120 MAIN PPO ENGINE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        pass
        
    revert_collision_meshes()

    print("🚀 Initializing THE SIM-TO-REAL APEX PREDATOR (V120 1v1 DUELIST)...")
    
    try:
        temp_env = build_env()
        dummy_reset = temp_env.reset()
        if isinstance(dummy_reset, tuple): dummy_reset = dummy_reset[0]
        obs_size = np.atleast_2d(dummy_reset).shape[-1]
        
        act_size = temp_env.action_space.n 
        temp_env.close()
        print(f"✅ Domain Randomization Env Built! True 1v1 Obs Size: {obs_size} (Padding Safe!) | Optimized Actions: {act_size}")
    except Exception as e:
        print(f"🚨 FATAL: build_env() crashed!\n{traceback.format_exc()}")
        sys.exit(1)

    WORKER_CORES = min(40, mp.cpu_count()) 
    
    GLOBAL_BATCH_SIZE = 100_000 
    EXP_BUFFER = 100_000 
    MINI_BATCH = 50_000 
    
    # 🛑 SOTA V120 FIX: 5k Deep Learning Horizons Activated
    BASE_ITERS = 5000
    EXTENSION_STEP = 2000
    TOTAL_ITERS = BASE_ITERS
    
    learner = Learner(
        build_env,
        n_proc=WORKER_CORES, 
        ppo_batch_size=GLOBAL_BATCH_SIZE,
        ts_per_iteration=GLOBAL_BATCH_SIZE,
        exp_buffer_size=EXP_BUFFER, 
        ppo_minibatch_size=MINI_BATCH, 
        ppo_ent_coef=0.01,
        
        standardize_obs=False,
        standardize_returns=True,
        
        policy_lr=2e-4,
        critic_lr=4e-4, 
        
        ppo_epochs=3, 
        
        policy_layer_sizes=(512, 512, 512),               
        critic_layer_sizes=(512, 512, 512),      
        
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_to_wandb=False
    )

    # 🛑 ♻️ THE ULTIMATE AUTO-RESUME PROTOCOL ♻️ 🛑
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

            possible_ckpt_names = [f"ckpt_V{v}_{start_iter}" for v in range(130, 20, -1)] + [f"ckpt_{start_iter}"]
            ckpt_path = None
            for name in possible_ckpt_names:
                if os.path.exists(os.path.join(ckpt_dir, name)):
                    ckpt_path = os.path.join(ckpt_dir, name)
                    break
                    
            raw_pt_path = os.path.join(ckpt_dir, f"raw_policy_weights_{start_iter}.pt")
            loaded = False
            
            try:
                try: policy_net = learner.ppo_learner.policy
                except AttributeError: policy_net = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
                device = next(policy_net.parameters()).device

                if ckpt_path and os.path.exists(os.path.join(ckpt_path, "PPO_POLICY.pt")):
                    try:
                        try:
                            learner.load(ckpt_path, load_wandb=False)
                        except TypeError:
                            learner.load(ckpt_path) 
                        print(f"   ✅ NATIVE LOAD SUCCESS: Loaded full PyTorch brain from {ckpt_path}")
                        loaded = True
                    except Exception as e:
                        print(f"   ⚠️ Native load failed. Attempting manual injection...")
                        try:
                            policy_net.load_state_dict(torch.load(os.path.join(ckpt_path, "PPO_POLICY.pt"), map_location=device), strict=False)
                            try: learner.ppo_learner.value_net.load_state_dict(torch.load(os.path.join(ckpt_path, "PPO_VALUE_NET.pt"), map_location=device), strict=False)
                            except: pass
                            
                            if os.path.exists(os.path.join(ckpt_path, "REWARD_STANDARDIZER.pt")) and hasattr(learner.ppo_learner, "reward_standardizer"):
                                try: learner.ppo_learner.reward_standardizer.load_state_dict(torch.load(os.path.join(ckpt_path, "REWARD_STANDARDIZER.pt"), map_location=device))
                                except: pass
                            
                            bk_path = os.path.join(ckpt_path, "BOOK_KEEPING_VARS.json")
                            if os.path.exists(bk_path):
                                with open(bk_path, 'r') as f:
                                    bk_vars = json.load(f)
                                    if "cumulative_timesteps" in bk_vars:
                                        learner.agent.cumulative_timesteps = bk_vars["cumulative_timesteps"]
                                        
                            print(f"   ✅ Manually Restored PyTorch Brain from folder.")
                            loaded = True
                        except Exception as e_man:
                            print(f"   ⚠️ Manual load failed: {e_man}")

                if not loaded and os.path.exists(raw_pt_path):
                    try:
                        policy_net.load_state_dict(torch.load(raw_pt_path, map_location=device), strict=False)
                        print(f"   ✅ Restored Neural Network Actor Brain from {raw_pt_path}")
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
            
            torch.set_num_threads(1)
            experience, metrics, steps, coll_time = learner.agent.collect_timesteps(GLOBAL_BATCH_SIZE)
            
            learner.add_new_experience(experience)
            
            torch.set_num_threads(WORKER_CORES) 
            learner.ppo_learner.learn(learner.experience_buffer)
            learner.experience_buffer.clear()
            
            learner.agent.cumulative_timesteps += steps
            
            progress = min(1.0, i / max(1, TOTAL_ITERS))
            new_policy_lr = 2e-4 - ((2e-4 - 1e-5) * progress)
            new_critic_lr = 4e-4 - ((4e-4 - 5e-5) * progress) 
            new_ent = 0.01 - ((0.01 - 0.005) * progress)
            
            try:
                if hasattr(learner.ppo_learner, 'optimizer'):
                    for param_group in learner.ppo_learner.optimizer.param_groups: 
                        param_group['lr'] = new_policy_lr
                else:
                    for param_group in learner.ppo_learner.policy_optimizer.param_groups: param_group['lr'] = new_policy_lr
                    for param_group in learner.ppo_learner.value_optimizer.param_groups: param_group['lr'] = new_critic_lr
            except Exception:
                pass
            
            learner.ppo_ent_coef = new_ent
            learner.ppo_learner.ent_coef = new_ent

            if (i + 1) > start_iter and (i + 1) % 500 == 0:
                print(f"\n💾 Initiating Cloud Backup for Iteration {i+1}...")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                ckpt_folder = os.path.join(ckpt_dir, f"ckpt_V130_{i+1}")
                os.makedirs(ckpt_folder, exist_ok=True)
                
                try:
                    learner.ppo_learner.save_to(ckpt_folder)
                    
                    bk_vars = {"cumulative_timesteps": int(learner.agent.cumulative_timesteps)}
                    with open(os.path.join(ckpt_folder, "BOOK_KEEPING_VARS.json"), "w") as f:
                        json.dump(bk_vars, f)
                        
                    with open(os.path.join(ckpt_folder, "config.json"), "w") as f:
                        json.dump({}, f)
                        
                    print(f"   ✅ Perfect Replica Checkpoint Secure: All files saved to {ckpt_folder}!")
                except Exception as e:
                    print(f"   ⚠️ Perfect Replica save failed: {e}")
                
                try:
                    try: policy_net = learner.ppo_learner.policy
                    except AttributeError: policy_net = getattr(learner, 'policy', getattr(learner, 'agent', learner)).actor
                    device_net = next(policy_net.parameters()).device
                    
                    fallback_path = os.path.join(ckpt_dir, f"raw_policy_weights_{i+1}.pt")    
                    torch.save(policy_net.state_dict(), fallback_path)
                    
                    onnx_path = os.path.join(ckpt_dir, f"SOTA_RLBot_V130_Iter_{i+1}.onnx")
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
        export_path_drive = os.path.join(save_dir, "SOTA_RLBot_V130_Final.onnx")
        export_path_fallback = "SOTA_RLBot_V130_FALLBACK.onnx"
        
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
            print(f"\nWARNING: Google Drive export failed! (Did the drive unmount?)")
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
