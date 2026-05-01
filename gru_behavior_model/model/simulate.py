"""
Nexu behavior simulation — generates (obs, action) training data.

Observation vector (16 floats, all in [0,1]):
  0  touch_head
  1  touch_body
  2  touch_tail
  3  collision_front
  4  collision_left
  5  collision_right
  6  person_visible
  7  person_distance      (1.0 = very close, 0.0 = far/absent)
  8  wake_word            (1.0 = just heard "Nexu")
  9  battery_level        (1.0 = full)
 10  time_since_touch     (1.0 = long ago → restless)
 11  idle_ticks_norm      (1.0 = been idle a long time)
 12  sound_level          (1.0 = very loud environment)
 13  motion_energy        (1.0 = Nexu was moving recently)
 14  personality_curiosity (fixed at boot, 0..1)
 15  personality_playful   (fixed at boot, 0..1)

Actions (12):
  0  idle_sit
  1  look_around
  2  approach_person
  3  follow_person
  4  play_gesture
  5  vocal_happy
  6  vocal_curious
  7  vocal_alert
  8  sleep
  9  avoid_obstacle
 10  self_groom
 11  wag_tail
"""

import random
import numpy as np
from pathlib import Path

OBS_SIZE    = 16
NUM_ACTIONS = 12
HIDDEN_SIZE = 64

# Action indices
IDLE_SIT       = 0
LOOK_AROUND    = 1
APPROACH       = 2
FOLLOW         = 3
PLAY_GESTURE   = 4
VOCAL_HAPPY    = 5
VOCAL_CURIOUS  = 6
VOCAL_ALERT    = 7
SLEEP          = 8
AVOID          = 9
SELF_GROOM     = 10
WAG_TAIL       = 11


class NexuEnv:
    """Simulates Nexu's world for one episode."""

    def __init__(self, personality_curiosity=0.7, personality_playful=0.6):
        self.curiosity = personality_curiosity
        self.playful   = personality_playful
        self.reset()

    def reset(self):
        self.tick              = 0
        self.person_visible    = False
        self.person_dist       = 0.0
        self.touch_head        = False
        self.touch_body        = False
        self.touch_tail        = False
        self.collision_front   = False
        self.collision_left    = False
        self.collision_right   = False
        self.wake_word         = False
        self.battery           = random.uniform(0.5, 1.0)
        self.time_since_touch  = random.uniform(0.0, 0.5)
        self.idle_ticks        = 0
        self.sound_level       = random.uniform(0.0, 0.3)
        self.motion_energy     = 0.0
        self.last_action       = IDLE_SIT

    def obs(self) -> np.ndarray:
        o = np.zeros(OBS_SIZE, dtype=np.float32)
        o[0]  = float(self.touch_head)
        o[1]  = float(self.touch_body)
        o[2]  = float(self.touch_tail)
        o[3]  = float(self.collision_front)
        o[4]  = float(self.collision_left)
        o[5]  = float(self.collision_right)
        o[6]  = float(self.person_visible)
        o[7]  = self.person_dist
        o[8]  = float(self.wake_word)
        o[9]  = self.battery
        o[10] = min(self.time_since_touch, 1.0)
        o[11] = min(self.idle_ticks / 100.0, 1.0)
        o[12] = self.sound_level
        o[13] = self.motion_energy
        o[14] = self.curiosity
        o[15] = self.playful
        return o

    def policy(self) -> int:
        """Hand-crafted Nexu personality policy with randomness."""
        # Collision takes priority
        if self.collision_front or self.collision_left or self.collision_right:
            return AVOID

        # Wake word: greet enthusiastically
        if self.wake_word:
            return random.choice([VOCAL_HAPPY, WAG_TAIL, APPROACH])

        # Touch responses
        if self.touch_head:
            return random.choice([VOCAL_HAPPY, WAG_TAIL, IDLE_SIT])
        if self.touch_body:
            return random.choice([WAG_TAIL, VOCAL_HAPPY])
        if self.touch_tail:
            return random.choice([VOCAL_ALERT, VOCAL_ALERT, LOOK_AROUND])

        # Person nearby
        if self.person_visible:
            if self.person_dist > 0.7:
                return random.choice([FOLLOW, FOLLOW, APPROACH, LOOK_AROUND])
            elif self.person_dist > 0.3:
                return random.choice([PLAY_GESTURE, VOCAL_HAPPY, FOLLOW, APPROACH])
            else:  # very close
                return random.choice([PLAY_GESTURE, WAG_TAIL, VOCAL_HAPPY, IDLE_SIT])

        # Low battery
        if self.battery < 0.2:
            return SLEEP

        # Long idle
        if self.idle_ticks > 60:
            if random.random() < self.curiosity:
                return LOOK_AROUND
            else:
                return random.choice([SELF_GROOM, SLEEP, IDLE_SIT])

        if self.idle_ticks > 30:
            return random.choice([LOOK_AROUND, SELF_GROOM, IDLE_SIT, VOCAL_CURIOUS])

        # Baseline personality-driven random behavior
        if random.random() < 0.05 * self.playful:
            return PLAY_GESTURE
        if random.random() < 0.03 * self.curiosity:
            return LOOK_AROUND

        return IDLE_SIT

    def step(self, action: int):
        """Advance environment one tick."""
        self.tick += 1
        self.wake_word = False  # one-shot

        # Drain battery slowly
        self.battery = max(0.0, self.battery - 0.0005)

        # Motion energy decay
        self.motion_energy *= 0.9
        if action in (APPROACH, FOLLOW, AVOID, PLAY_GESTURE):
            self.motion_energy = min(1.0, self.motion_energy + 0.3)

        # Touch decays
        self.touch_head  = False
        self.touch_body  = False
        self.touch_tail  = False
        self.collision_front = False
        self.collision_left  = False
        self.collision_right = False

        # Time since touch
        self.time_since_touch = min(1.0, self.time_since_touch + 0.01)

        # Idle accumulation
        if action == IDLE_SIT:
            self.idle_ticks += 1
        else:
            self.idle_ticks = max(0, self.idle_ticks - 5)

        self.last_action = action


def run_scenario_person_visit(env: NexuEnv, steps=120) -> list:
    """Person walks up, plays, pats Nexu, then leaves."""
    data = []
    env.reset()

    for i in range(steps):
        # Scenario timeline
        if i == 10:
            env.person_visible = True
            env.person_dist    = 0.1
        elif 10 < i < 40:
            env.person_dist = min(1.0, env.person_dist + 0.03)
        elif 40 <= i < 50:
            env.person_dist    = 1.0
            if i % 5 == 0:
                env.touch_head = True
                env.time_since_touch = 0.0
            if i == 47:
                env.touch_tail = True  # accidental tail touch → alert
        elif 50 <= i < 80:
            env.person_dist = max(0.0, env.person_dist - 0.04)
            if i == 60:
                env.person_visible = False
        elif i == 15:
            env.wake_word = True

        obs    = env.obs()
        action = env.policy()
        data.append((obs.copy(), action))
        env.step(action)

    return data


def run_scenario_idle_exploration(env: NexuEnv, steps=150) -> list:
    """Nexu alone — idle, grooms, looks around."""
    data = []
    env.reset()
    env.person_visible = False

    for i in range(steps):
        if i % 40 == 25:
            env.sound_level = random.uniform(0.5, 1.0)
        else:
            env.sound_level = max(0.0, env.sound_level - 0.05)

        obs    = env.obs()
        action = env.policy()
        data.append((obs.copy(), action))
        env.step(action)

    return data


def run_scenario_collision(env: NexuEnv, steps=60) -> list:
    """Nexu encounters obstacles while exploring."""
    data = []
    env.reset()

    for i in range(steps):
        if i % 15 == 0:
            which = random.choice(['front', 'left', 'right'])
            if which == 'front':  env.collision_front = True
            elif which == 'left': env.collision_left  = True
            else:                 env.collision_right = True

        obs    = env.obs()
        action = env.policy()
        data.append((obs.copy(), action))
        env.step(action)

    return data


def run_scenario_low_battery(env: NexuEnv, steps=80) -> list:
    """Battery drains — Nexu gradually winds down."""
    data = []
    env.reset()
    env.battery = 0.25

    for i in range(steps):
        env.battery = max(0.0, env.battery - 0.003)
        obs    = env.obs()
        action = env.policy()
        data.append((obs.copy(), action))
        env.step(action)

    return data


def generate_dataset(n_episodes=500, seed=42) -> tuple:
    random.seed(seed)
    np.random.seed(seed)

    all_obs, all_actions = [], []
    scenarios = [
        run_scenario_person_visit,
        run_scenario_idle_exploration,
        run_scenario_collision,
        run_scenario_low_battery,
    ]

    for ep in range(n_episodes):
        curiosity = np.clip(random.gauss(0.7, 0.15), 0.1, 1.0)
        playful   = np.clip(random.gauss(0.6, 0.15), 0.1, 1.0)
        env = NexuEnv(curiosity, playful)
        scenario_fn = random.choice(scenarios)
        steps = random.randint(60, 200)
        episode = scenario_fn(env, steps=steps)
        for obs, action in episode:
            all_obs.append(obs)
            all_actions.append(action)

    X = np.array(all_obs,    dtype=np.float32)
    y = np.array(all_actions, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # 90/10 split
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    print("Generating Nexu behavior dataset...")
    X_train, y_train, X_val, y_val = generate_dataset(n_episodes=600)

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    DATA.mkdir(exist_ok=True)

    np.save(str(DATA / "X_train.npy"), X_train)
    np.save(str(DATA / "y_train.npy"), y_train)
    np.save(str(DATA / "X_val.npy"),   X_val)
    np.save(str(DATA / "y_val.npy"),   y_val)

    print(f"Train: {len(X_train)} steps  Val: {len(y_val)} steps")
    print(f"Action distribution (train):")
    actions = ["idle_sit","look_around","approach","follow","play_gesture",
               "vocal_happy","vocal_curious","vocal_alert","sleep","avoid","self_groom","wag_tail"]
    for i, name in enumerate(actions):
        count = (y_train == i).sum()
        print(f"  {i:2d} {name:15s}: {count:6d} ({100*count/len(y_train):.1f}%)")

    # Save calibration samples
    CAL = ROOT / "calibration_data" / "samples"
    CAL.mkdir(parents=True, exist_ok=True)
    cal_idx = np.random.choice(len(X_train), min(200, len(X_train)), replace=False)
    zeros_hidden = np.zeros((1, HIDDEN_SIZE), dtype=np.float32)
    for i, idx in enumerate(cal_idx):
        np.savez(str(CAL / f"sample_{i:04d}.npz"),
                 obs=X_train[idx:idx+1, np.newaxis, :],  # [1,1,16]
                 hidden_in=zeros_hidden)
    print(f"Saved {len(cal_idx)} calibration samples to {CAL}")
    print("Done. Run train.py next.")
