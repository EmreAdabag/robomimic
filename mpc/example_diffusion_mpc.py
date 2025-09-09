#!/usr/bin/env python
"""
Simple example showing how to use Diffusion Policy with MPC integration.

This example demonstrates the minimal setup needed to train and use
a diffusion policy with your differentiable MPC layer.
"""
import torch
import numpy as np
import os

# robomimic imports
from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig
from robomimic.algo.diffusion_policy import DiffusionPolicyMPC
from robomimic.utils.dataset import SequenceDataset

# Your MPC dynamics
from panda_mpc_simple import PandaDynamics


def create_mpc_enhanced_policy(dataset_path, urdf_path):
    """
    Create a diffusion policy with MPC integration.

    Args:
        dataset_path: Path to your training dataset
        urdf_path: Path to Panda URDF file
    """
    # Create config with MPC enabled
    config = DiffusionPolicyConfig()
    config.algo.mpc.enabled = True
    config.algo.mpc.horizon = 20

    # Ensure actions are normalized for diffusion
    config.train.action_config = {
        "actions": {"normalization": "min_max"}
    }

    # Load dataset
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=config.all_obs_keys,
        dataset_keys=config.train.dataset_keys,
        action_keys=config.train.action_keys,
        load_next_obs=config.train.hdf5_load_next_obs,
        frame_stack=config.train.frame_stack,
        seq_length=config.train.seq_length,
        pad_frame_stack=config.train.pad_frame_stack,
        pad_seq_length=config.train.pad_seq_length,
        get_pad_mask=False,
        goal_mode=config.train.goal_mode,
        hdf5_cache_mode=config.train.hdf5_cache_mode,
        hdf5_use_swmr=config.train.hdf5_use_swmr,
        hdf5_normalize_obs=config.train.hdf5_normalize_obs,
        filter_by_attribute=config.train.hdf5_filter_key,
    )

    # Create your differentiable dynamics model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_model = PandaDynamics(
        urdf_path=urdf_path,
        dt=0.1,  # 100ms timestep
        device=device
    )

    # Create MPC-enhanced policy
    policy = DiffusionPolicyMPC(
        algo_config=config.algo,
        obs_config=config.observation,
        global_config=config,
        obs_key_shapes=dataset.obs_key_shapes,
        ac_dim=dataset.ac_dim,
        device=device,
        dynamics_model=dynamics_model,
        mpc_horizon=20,
        mpc_enabled=True
    )

    return policy, dataset, config


def train_policy(policy, dataset, config, num_epochs=100):
    """
    Train the policy using standard robomimic training loop.
    """
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # Training loop
    policy.set_train()

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(data_loader):
            # Process batch
            input_batch = policy.process_batch_for_training(batch)

            # Get normalization stats
            obs_normalization_stats = dataset.get_obs_normalization_stats() if config.train.hdf5_normalize_obs else None
            input_batch = policy.postprocess_batch_for_training(
                input_batch,
                obs_normalization_stats=obs_normalization_stats
            )

            # Training step (gradients flow through MPC during inference)
            info = policy.train_on_batch(input_batch, epoch, validate=False)
            epoch_losses.append(info["losses"]["total"])

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {info['losses']['total']:.4f}")

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        # Save model periodically
        if (epoch + 1) % 20 == 0:
            torch.save(policy.serialize(), f"diffusion_mpc_model_epoch_{epoch+1}.pth")
            print(f"Saved model at epoch {epoch+1}")


def test_policy(policy):
    """
    Test the trained policy - MPC refinement will be applied during inference.
    """
    policy.set_eval()
    policy.reset()  # Reset action queue

    # Create dummy observation (customize based on your obs structure)
    obs_dict = {
        'robot0_joint_pos': torch.zeros(1, 7),  # Joint positions
        'robot0_joint_vel': torch.zeros(1, 7),  # Joint velocities
        # Add other observation keys as needed
    }

    with torch.no_grad():
        # Get action - will use diffusion + MPC
        action = policy.get_action(obs_dict)
        print(f"Generated action shape: {action.shape}")
        print(f"Action values: {action.squeeze().cpu().numpy()}")


def main():
    # Paths - update these for your setup
    dataset_path = "path/to/your/dataset.hdf5"
    urdf_path = "/home/emrea/panda_pytorch/robot_description/panda_with_gripper.urdf"

    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please update dataset_path with your actual dataset")
        return

    if not os.path.exists(urdf_path):
        print(f"URDF not found: {urdf_path}")
        print("Please update urdf_path with your actual URDF file")
        return

    print("Creating MPC-enhanced diffusion policy...")
    policy, dataset, config = create_mpc_enhanced_policy(dataset_path, urdf_path)

    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Action dimension: {dataset.ac_dim}")
    print(f"MPC enabled: {policy.mpc_enabled}")

    # Train the policy
    print("\nStarting training...")
    train_policy(policy, dataset, config, num_epochs=50)

    # Test the policy
    print("\nTesting trained policy...")
    test_policy(policy)

    print("Done!")


if __name__ == "__main__":
    main()
