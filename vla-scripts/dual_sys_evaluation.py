from collections import deque
import logging
import os
import time

import numpy as np
from PIL import Image
import torch
from einops import rearrange

from calvin_agent.models.calvin_base_model import CalvinBaseModel

logger = logging.getLogger(__name__)


def get_openvla_prompt(instruction: str, tokenized_action: str = None) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class DualSystemCalvinEvaluation(CalvinBaseModel):
    def __init__(self, model, processor, action_tokenizer):
        super().__init__()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = processor
        self.dual_sys = model
        self.action_tokenizer = action_tokenizer

        self.temporal_size = 8
        self.temporal_mask = torch.flip(torch.triu(torch.ones(self.temporal_size, self.temporal_size, dtype=torch.bool)), dims=[1]).numpy()
        
        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)

        self.action = None
        self.hidden_states = None
        self.obs_buffer = None

        # Action chunking with temporal aggregation
        balancing_factor = 0.1
        self.temporal_weights = np.array([np.exp(-1 * balancing_factor * i) for i in range(self.temporal_size)])[:, None]

        # Dataset statics (rougnly computed with 10k samples in CALVIN)
        self.depth_max = 6.2
        self.depth_min = 3.5
        self.gripper_depth_max = 2.0
        self.gripper_depth_min = 0

        self.hist_action = deque(maxlen=4)
        self.last_step_profile = {}
        self._fast_device = None

    def _slow_system(self):
        return self.dual_sys.module.slow_system if hasattr(self.dual_sys, "module") else self.dual_sys.slow_system

    def _dual_system(self):
        return self.dual_sys.module if hasattr(self.dual_sys, "module") else self.dual_sys

    def _runtime_device(self):
        slow_system = self._slow_system()
        if hasattr(slow_system, "device"):
            return slow_system.device
        return next(slow_system.parameters()).device

    def _runtime_dtype(self):
        slow_system = self._slow_system()
        if hasattr(slow_system, "dtype") and slow_system.dtype is not None:
            return slow_system.dtype
        return next(slow_system.parameters()).dtype

    def _fast_system(self):
        return self._dual_system().ema_fast_system.ema_model

    def _ensure_fast_system_device(self, runtime_device):
        if self._fast_device == runtime_device:
            return
        fast_system = self._fast_system()
        fast_system.to(runtime_device)
        fast_system.eval()
        self._fast_device = runtime_device
        logger.info("Moved fast system to %s for evaluation", runtime_device)

        
    def reset(self,):
        """
        This is called
        """

        self.action_buffer = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0], 7))
        self.action_buffer_mask = np.zeros((self.temporal_mask.shape[0], self.temporal_mask.shape[0]), dtype=np.bool_)
        self.obs_buffer = None
        self.hist_action.clear()
        self.last_step_profile = {}


    def step(self, obs, instruction, step):
        """
        Args:
            obs: environment observations
            instruction: embedded language goal
        Returns:
            action: predicted action
        """
        profile = {
            "step": int(step),
            "slow_system": False,
        }
        total_start = time.perf_counter()
        live_profile = os.environ.get("ROBODUAL_PROFILE_LIVE") == "1"

        with torch.inference_mode():
            preprocess_start = time.perf_counter()
            image = obs["rgb_obs"]['rgb_static']
            gripper_image = obs["rgb_obs"]['rgb_gripper']
            runtime_device = self._runtime_device()
            runtime_dtype = self._runtime_dtype()
            self._ensure_fast_system_device(runtime_device)
            gripper_image = self.processor.image_processor.apply_transform(Image.fromarray(gripper_image))[:3].unsqueeze(0).to(runtime_device)

            tactile_image = None
            # tactile_image = torch.from_numpy(obs["rgb_obs"]['rgb_tactile']).permute(2,0,1).unsqueeze(0).to(runtime_device, dtype=torch.float) / 255
            depth_image = (torch.as_tensor(obs["depth_obs"]['depth_static'], device=runtime_device).unsqueeze(0) - self.depth_min) / (self.depth_max - self.depth_min)
            depth_gripper = (torch.as_tensor(obs["depth_obs"]['depth_gripper'], device=runtime_device).unsqueeze(0) - self.gripper_depth_min) / (self.gripper_depth_max - self.gripper_depth_min)

            prompt = get_openvla_prompt(instruction)
            inputs = self.processor(prompt, Image.fromarray(image)).to(runtime_device, dtype=runtime_dtype)
            profile["preprocess_s"] = round(time.perf_counter() - preprocess_start, 4)
            profile["runtime_dtype"] = str(runtime_dtype)
            if live_profile:
                print(f"[live-profile] step={step} preprocess_done {profile}", flush=True)

            if (step + 1) % 8 == 0 or step == 0:
                slow_start = time.perf_counter()
                action, hidden_states = self._slow_system().predict_action(**inputs, do_sample=False)
                action = torch.as_tensor(action, device=hidden_states.device).unsqueeze(0)
                action = rearrange(action, 'b (f d) -> b f d', f=8)
                self.action = action[:, :, :7]
                self.hidden_states = hidden_states
                profile["slow_system"] = True
                profile["slow_system_s"] = round(time.perf_counter() - slow_start, 4)
                if live_profile:
                    print(f"[live-profile] step={step} slow_done {profile}", flush=True)

            num_cond_actions = 8 - (step + 1) % 8
            if step == 0:
                num_cond_actions = 8

            zero_actions = torch.zeros((1, self.temporal_size, 7), device=self.action.device)
            zero_actions[:, :num_cond_actions] = self.action[:, -num_cond_actions:]
            ref_actions = zero_actions

            state = torch.as_tensor(obs['robot_obs'], device=runtime_device, dtype=torch.float32)
            state = torch.cat([state[:6], state[[-1]]], dim=-1).unsqueeze(0)

            if step == 0:
                self.obs_buffer = image

            prev_img = self.processor.image_processor.apply_transform(Image.fromarray(self.obs_buffer))[:3].unsqueeze(0).to(runtime_device)
            obs = (inputs["pixel_values"][:, :3].to(torch.float32), prev_img)

            hist_action = torch.zeros((1, 4, 7), device=runtime_device)
            if self.hist_action:
                hist_stack = torch.stack(list(self.hist_action), dim=0).unsqueeze(0).to(runtime_device)
                hist_action[:, -hist_stack.shape[1]:] = hist_stack

            fast_start = time.perf_counter()
            dp_action = self._fast_system().predict_action(
                ref_action=ref_actions.to(torch.float32),
                action_cond=self.hidden_states.to(torch.float32),
                obs=obs,
                depth_obs=depth_image,
                gripper_obs=(gripper_image, depth_gripper),
                tactile_obs=tactile_image,
                lang=instruction,
                proprio=state,
                hist_action=hist_action,
            )
            profile["fast_system_s"] = round(time.perf_counter() - fast_start, 4)
            if live_profile:
                print(f"[live-profile] step={step} fast_done {profile}", flush=True)
        self.obs_buffer = image
        to_numpy_start = time.perf_counter()
        action = dp_action.detach().to(torch.float32).cpu().numpy()
        profile["to_numpy_s"] = round(time.perf_counter() - to_numpy_start, 4)


        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.temporal_mask

        # Add to action buffer
        self.action_buffer[0] = action  
        self.action_buffer_mask[0] = np.array([True] * self.temporal_mask.shape[0], dtype=np.bool_)

        # Ensemble temporally to predict action
        action_prediction = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1] * self.temporal_weights, axis=0) / np.sum(self.action_buffer_mask[:, 0:1] * self.temporal_weights)


        if action_prediction[-1] < -0.5:
            action_prediction[-1] = -1
        else:
            action_prediction[-1] = 1

        self.hist_action.append(torch.from_numpy(action_prediction).to(torch.float32))
        profile["total_s"] = round(time.perf_counter() - total_start, 4)
        self.last_step_profile = profile
        if live_profile:
            print(f"[live-profile] step={step} total_done {profile}", flush=True)

        return action_prediction
