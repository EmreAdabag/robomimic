# Simple Diffusion Policy + MPC Integration

This is a minimal integration that adds differentiable MPC refinement to robomimic's Diffusion Policy with only 3 small changes to existing code.

## What it does

1. **Training**: Uses standard diffusion policy training (no MPC during training)
2. **Inference**: After diffusion generates a trajectory, MPC refines it using dynamics constraints
3. **Gradients**: MPC refinement happens within the computational graph, so gradients flow through during inference

## Changes Made

### 1. Added MPC-enhanced policy class (`robomimic/algo/diffusion_policy.py`)
```python
class DiffusionPolicyMPC(DiffusionPolicyUNet):
    """Diffusion Policy with MPC refinement after trajectory generation"""
```

### 2. Added MPC config option (`robomimic/config/diffusion_policy_config.py`)
```python
# MPC parameters
self.algo.mpc = {}
self.algo.mpc.enabled = False  # Set to True to enable
self.algo.mpc.horizon = 20
```

### 3. Updated algorithm factory to use MPC variant when enabled

## Usage

### Basic Setup
```python
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig
from robomimic.algo.diffusion_policy import DiffusionPolicyMPC
from mpc.panda_mpc_simple import PandaDynamics

# Create config with MPC enabled
config = DiffusionPolicyConfig()
config.algo.mpc.enabled = True

# Create your dynamics model
dynamics = PandaDynamics(urdf_path="panda.urdf", dt=0.1, device="cuda")

# Create policy
policy = DiffusionPolicyMPC(
    dynamics_model=dynamics,
    mpc_enabled=True,
    **standard_args
)
```

### Training (uses existing robomimic scripts)
```bash
# Train with standard robomimic - just set mpc.enabled=True in config
python robomimic/scripts/train.py --config config.json
```

### Inference
```python
# MPC refinement automatically applied during get_action()
action = policy.get_action(obs_dict)
```

## Integration Points

The MPC layer plugs into `_get_action_trajectory()` method:
1. Diffusion generates initial trajectory 
2. Extract robot state from observations
3. Apply MPC refinement using your `PandaDynamics` 
4. Return refined trajectory

## Customization

### State Extraction
Modify `_extract_state_from_obs()` for your observation structure:
```python
def _extract_state_from_obs(self, obs_dict):
    # Customize based on your obs keys
    pos = obs_dict['robot0_joint_pos']  # [B, 7]
    vel = obs_dict['robot0_joint_vel']  # [B, 7] 
    return torch.cat([pos, vel], dim=-1)  # [B, 14]
```

### MPC Refinement
Modify `_apply_mpc_refinement()` to use your specific MPC setup:
```python
def _apply_mpc_refinement(self, initial_actions, obs_dict):
    # Add your MPC optimization logic here
    # Can use QuadCost, your existing MPC controller, etc.
    return refined_actions
```

## Requirements

- Your existing `PandaDynamics` class
- MPC.pytorch (optional, for more sophisticated MPC)
- Standard robomimic dependencies

## Files

- `robomimic/algo/diffusion_policy.py` - Main integration (DiffusionPolicyMPC class)
- `robomimic/config/diffusion_policy_config.py` - Config options
- `mpc/example_diffusion_mpc.py` - Usage example
- `mpc/panda_mpc_simple.py` - Your existing dynamics model

That's it! Minimal changes, maximum compatibility with existing robomimic workflows.