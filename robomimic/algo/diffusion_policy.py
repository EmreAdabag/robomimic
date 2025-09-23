"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi

FIXES APPLIED:
- Fixed division by zero in MPC loss calculation when no examples converge (lines 308-312, 393-397)
- Cleaned up conditional loss mode selection for better maintainability
- Added proper handling of infinite losses to prevent training instability
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.models.diffusion_policy_nets as DPNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from pandadotpytorch.mpcpanda import PandaEETrackingMPCLayer
# torch.autograd.set_detect_anomaly(True)

@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()


class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = DPNets.ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            "policy": nn.ModuleDict({
                "obs_encoder": obs_encoder,
                "noise_pred_net": noise_pred_net
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None
        
        self.mpc_timestep = 0.01
        self.mpc_timesteps_per_timestep = torch.tensor(0.05 / self.mpc_timestep, dtype=torch.int32)
        self.mpclayer = PandaEETrackingMPCLayer(
            urdf_path="/home/emrea/panda_pytorch/robot_description/panda_with_gripper.urdf",
            T=40,
            dt=self.mpc_timestep,
            device=self.device,
            with_gravity=True,
            lqr_iter=10,
            eps=1.e-2,
            verbose=0,
        ).to(self.device)

        self.qweight = torch.tensor(4.0, device=self.device) 
        self.vweight = torch.tensor(1e-2, device=self.device) 
        self.uweight = torch.tensor(1e-6, device=self.device)
    
    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError("'actions' must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.")
            self.action_check_done = True
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)
        
    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        B = batch["actions"].shape[0]
        
        
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch["actions"]
            
            # encode obs
            inputs = {
                "obs": batch["obs"],
                "goal": batch["goal_obs"]
            }
            for k in self.obs_shapes:
                # first two dimensions should be [B, T] for inputs
                assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
            
            obs_features = TensorUtils.time_distributed(inputs, self.nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]

            obs_cond = obs_features.flatten(start_dim=1)
            
            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)
            
            # predict the noise residual
            noise_pred = self.nets["policy"]["noise_pred_net"](
                noisy_actions, timesteps, global_cond=obs_cond)
            
            imitation_loss = torch.zeros(1, device=self.device).squeeze(0)
            post_mpc_loss = torch.zeros(1, device=self.device).squeeze(0)
            scale = torch.tensor([0.2806792 , 0.41592544, 0.18729036, 0.42120016, 0.20262541, 0.37093928, 0.7342546, 0.01280932, 0.0115944 ], device=self.device)
            offset = torch.tensor([-0.03401133,  0.56654316,  0.04472703, -2.2583189 , -0.01904401, 3.0475502 ,  0.67210728, 0.02740401, -0.02882084], device=self.device)

            # Loss mode selection - set exactly one to True
            use_sparse_mpc_loss = False
            use_dense_mpc_loss = False  
            use_action_loss = False
            use_sparse_noise_loss = False
            use_base_noise_loss = True
            
            if use_sparse_mpc_loss:
                # noise_pred [128, 16, 9], [B, action_horizon, joints]

                # remove noise
                # breakpoint()
                alpha_bar_t = self.noise_scheduler.alphas_cumprod[timesteps].to(noisy_actions.dtype)  # (B,)
                sqrt_alpha_bar_t = alpha_bar_t.sqrt().view(-1, 1, 1)
                sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1)

                # x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
                x0_hat = (noisy_actions - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                actpred = x0_hat
                # actpred = actions + (noise - noise_pred)

                # unnormalize
                actpred = actpred * scale + offset
                actions_unscaled = actions * scale + offset

                goal_indices = torch.arange(Tp, device=self.device)[1::2]
                
                # Apply correct weighting to make it equivalent to noise loss
                weight = alpha_bar_t / (1.0 - alpha_bar_t)  # (B,)
                weight = weight.view(-1, 1, 1)  # (B, 1, 1)
                
                imitation_loss_unweighted = F.mse_loss(
                    actpred[:, goal_indices, :7],
                    actions_unscaled[:, goal_indices, :7],
                    reduction='none'
                )
                imitation_loss = (imitation_loss_unweighted * weight[:, :len(goal_indices), :]).mean()


                jointgoals = actpred[:,goal_indices,:7] # [B, K, joints]
                goal_timesteps = goal_indices * self.mpc_timesteps_per_timestep # 1 + torch.arange(jointgoals.shape[1]) * self.mpc_timesteps_per_timestep

                # cheating_jointgoals = actions_unscaled[:,:,:7]
                
                # batch['obs']['robot0_joint_pos'] torch.Size([128, 2, 7])
                qcur = batch['obs']['robot0_joint_pos'][:, -1, :]   # [B, history, joints]
                vcur = batch['obs']['robot0_joint_vel'][:, -1, :]   # [B, history, joints]
                xcur = torch.hstack([qcur, vcur])

                # breakpoint()
                x_mpc, u_mpc, obj, converged_mask = self.mpclayer(xcur, jointgoals,  goal_timesteps, self.qweight, self.vweight, self.uweight)

                # mask bad batches with zeros
                mask_b = converged_mask.view(-1,1,1)
                x_mpc_eff = torch.where(mask_b, x_mpc, torch.zeros_like(x_mpc))
                u_mpc_eff = torch.where(mask_b, u_mpc, torch.zeros_like(u_mpc))
                x_mpc = torch.where(mask_b, x_mpc, torch.zeros_like(x_mpc))
                u_mpc = torch.where(mask_b, u_mpc, torch.zeros_like(u_mpc))


                action_indices = torch.arange(x_mpc.shape[1], device=self.device)[self.mpc_timesteps_per_timestep::self.mpc_timesteps_per_timestep] # (torch.arange(x_mpc.shape[0] - 1) + 1)[::self.mpc_timesteps_per_timestep]
                # breakpoint()
                
                # allacts_pred = torch.cat([jointact_out, actpred[idx, :, -2:]], dim=2)
                # Mask out batches with NaNs/Infs BEFORE loss so backward never touches NaNs
                pred = x_mpc_eff[:, action_indices, :7]            # [B_good, H, 7]
                targ = actions_unscaled[:, :len(action_indices), :7]          # [B, H, 7]

                # Replace bad batches with zeros (constants) so grads are killed and values are finite
                per_batch_loss = F.mse_loss(pred, targ, reduction='none')  # [B, H, 7]
                
                # Apply the same weighting for mathematical equivalence to noise loss
                weight_mpc = weight[:, :per_batch_loss.shape[1], :per_batch_loss.shape[2]]  # [B, H, 7]
                weighted_loss = per_batch_loss * weight_mpc  # [B, H, 7]
                per_batch = weighted_loss.mean(dim=(1, 2))  # [B]
                num_converged = converged_mask.sum()
                if num_converged > 0:
                    post_mpc_loss = per_batch.sum() / num_converged
                else:
                    post_mpc_loss = torch.tensor(0.0, device=self.device)

                pre_weight = 0.1
                post_weight = 0.9
                loss = post_weight * post_mpc_loss
                print(f'imitation_loss: {imitation_loss}, post_mpc_loss: {post_mpc_loss}')
            elif use_action_loss:
                # base action loss (mathematically equivalent to noise loss)
                alpha_bar_t = self.noise_scheduler.alphas_cumprod[timesteps].to(noisy_actions.dtype)  # (B,)
                sqrt_alpha_bar_t = alpha_bar_t.sqrt().view(-1, 1, 1)
                sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1)

                # x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)
                x0_hat = (noisy_actions - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                actpred = x0_hat

                actpred = actpred * scale + offset
                actions_unscaled = actions * scale + offset

                # Apply correct weighting to make it equivalent to noise loss
                # The mathematical relationship is: action_loss = [(1 - alpha_bar_t) / alpha_bar_t] * noise_loss
                # So to get equivalent loss, we need to weight by alpha_bar_t / (1 - alpha_bar_t)
                weight = alpha_bar_t / (1.0 - alpha_bar_t)  # (B,)
                weight = weight.view(-1, 1, 1)  # (B, 1, 1)
                
                loss = F.mse_loss(actpred, actions_unscaled, reduction='none')  # (B, T, A)
                loss = (loss * weight).mean()  # Apply weighting and reduce
            elif use_dense_mpc_loss:
                # dense mpc action loss
                alpha_bar_t = self.noise_scheduler.alphas_cumprod[timesteps].to(noisy_actions.dtype)  # (B,)
                sqrt_alpha_bar_t = alpha_bar_t.sqrt().view(-1, 1, 1)
                sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1)

                actpred = (noisy_actions - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                actpred = actpred * scale + offset
                actions_unscaled = actions * scale + offset

                
                # Apply correct weighting to make it equivalent to noise loss
                weight = alpha_bar_t / (1.0 - alpha_bar_t)  # (B,)
                weight = weight.view(-1, 1, 1)  # (B, 1, 1)
                
                imitation_loss_unweighted = F.mse_loss(
                    actpred[:,:, :7],
                    actions_unscaled[:,:,:7],
                    reduction='none'
                )
                imitation_loss = (imitation_loss_unweighted * weight).mean()

                jointgoals = actpred[:,:,:7] # [B, K, joints]

                goal_timesteps = torch.tensor([self.mpc_timesteps_per_timestep * (i + 1) for i in range(actpred.shape[1])], device=self.device)
                # cheating_jointgoals = actions_unscaled[:,:,:7]
                
                # batch['obs']['robot0_joint_pos'] torch.Size([128, 2, 7])
                qcur = batch['obs']['robot0_joint_pos'][:, -1, :]   # [B, history, joints]
                vcur = batch['obs']['robot0_joint_vel'][:, -1, :]   # [B, history, joints]
                xcur = torch.hstack([qcur, vcur])

                x_mpc, u_mpc, obj, converged_mask = self.mpclayer(xcur, jointgoals,  goal_timesteps, self.qweight, self.vweight, self.uweight)

                # mask bad batches with zeros
                mask_b = converged_mask.view(-1,1,1)
                x_mpc_eff = torch.where(mask_b, x_mpc, torch.zeros_like(x_mpc))
                u_mpc_eff = torch.where(mask_b, u_mpc, torch.zeros_like(u_mpc))
                x_mpc = torch.where(mask_b, x_mpc, torch.zeros_like(x_mpc))
                u_mpc = torch.where(mask_b, u_mpc, torch.zeros_like(u_mpc))
                # Mask out batches with NaNs/Infs BEFORE loss so backward never touches NaNs
                
                action_indices = torch.arange(x_mpc_eff.shape[1], device=self.device)[::self.mpc_timesteps_per_timestep]
                
                pred = x_mpc_eff[:, action_indices, :7]            # [B_good, H, 7]
                targ = actions_unscaled[:, :len(action_indices), :7]          # [B, H, 7]

                # Replace bad batches with zeros (constants) so grads are killed and values are finite
                per_batch_loss = F.mse_loss(pred, targ, reduction='none')  # [B, H, 7]

                # Apply the same weighting for mathematical equivalence to noise loss
                weight_mpc = weight  # [B, H, 7]
                weighted_loss = per_batch_loss * weight_mpc  # [B, H, 7]
                per_batch = weighted_loss.mean(dim=(1, 2))  # [B]
                num_converged = converged_mask.sum()
                if num_converged > 0:
                    post_mpc_loss = per_batch.sum() / num_converged
                else:
                    post_mpc_loss = torch.tensor(0.0, device=self.device)

                pre_weight = 0.
                post_weight = 1.0
                loss = post_weight * post_mpc_loss + pre_weight * imitation_loss
                print(f'imitation_loss: {imitation_loss}, post_mpc_loss: {post_mpc_loss}')
            elif use_sparse_noise_loss:
                # sparse action loss
                goal_indices = torch.arange(Tp, device=self.device)[1::2] # (torch.arange(Tp - 1) + 1)[::self.mpc_timesteps_per_timestep]
                # breakpoint()
                loss = F.mse_loss(noise[:, goal_indices], noise_pred[:, goal_indices])
            elif use_base_noise_loss:
                # base noise loss
                loss = F.mse_loss(noise, noise_pred)
            else:
                raise ValueError("No loss mode selected! Set exactly one loss mode to True.")

            # logging
            losses = {
                # "pre_mpc_loss": imitation_loss,
                # "post_mpc_loss": post_mpc_loss,
                "l2_loss": loss
            }
            info["losses"] = TensorUtils.detach(losses)

            nanloss = loss.isnan().any()
            if nanloss:
                breakpoint()
                print("NANLOSS")

            if not validate and not nanloss:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                    max_grad_norm=1.0
                )
                print(f'grad norms: {policy_grad_norms}')
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    "policy_grad_norms": policy_grad_norms
                }
                info.update(step_info)

        return info
    
    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
    
    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        obs_queue = deque(maxlen=To)
        action_queue = deque(maxlen=Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue
    
    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """
        # obs_dict: key: [1,D]
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        
        if len(self.action_queue) == 0:
            # no actions left, run inference
            # [1,T,Da]
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)
            
            # put actions into the queue
            self.action_queue.extend(action_sequence[0])
        
        # has action, execute from left to right
        # [Da]

        scale = torch.tensor([0.2806792 , 0.41592544, 0.18729036, 0.42120016, 0.20262541, 0.37093928, 0.7342546, 0.01280932, 0.0115944 ], device=self.device)
        offset = torch.tensor([-0.03401133,  0.56654316,  0.04472703, -2.2583189 , -0.01904401, 3.0475502 ,  0.67210728, 0.02740401, -0.02882084], device=self.device)
        
        if 1:
            jointgoals = torch.stack(list(self.action_queue), dim=0)
            # jointgoals = thisaction.unsqueeze(0)
            jointgoals = jointgoals * scale + offset
            jointgoals = jointgoals[:,:7].unsqueeze(0) # [B, K, joints]
            goal_indices = torch.tensor([self.mpc_timesteps_per_timestep * (i + 1) for i in range(jointgoals.shape[1])])
            
            
            qcur = obs_dict['robot0_joint_pos'][:, -1, :] # [B, history, joints]
            vcur = obs_dict['robot0_joint_vel'][:, -1, :] # [B, history, joints]
            xcur = torch.hstack([qcur, vcur])

            x_mpc, u_mpc, obj, _ = self.mpclayer(
                xcur,
                jointgoals, 
                goal_indices,
                self.qweight,
                self.vweight,
                self.uweight
            )
            
            # res = thisaction * scale + offset
            # breakpoint()
            thisaction = self.action_queue.popleft() * scale + offset
            res = torch.hstack([x_mpc[0,5,:7], thisaction[-2:]])
        else:
            action = self.action_queue.popleft()
            joint_goal = action * scale + offset
            res = joint_goal
            
        # [1,Da]
        res = res.unsqueeze(0)
        return res
        
    def _get_action_trajectory(self, obs_dict, goal_dict=None):
        assert not self.nets.training
        To = self.algo_config.horizon.observation_horizon
        Ta = self.algo_config.horizon.action_horizon
        Tp = self.algo_config.horizon.prediction_horizon
        action_dim = self.ac_dim
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError
        
        # select network
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # encode obs
        inputs = {
            "obs": obs_dict,
            "goal": goal_dict
        }
        for k in self.obs_shapes:
            # first two dimensions should be [B, T] for inputs
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                # adding time dimension if not present -- this is required as
                # frame stacking is not invoked when sequence length is 1
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])
        obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, T, D]
        B = obs_features.shape[0]

        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = obs_features.flatten(start_dim=1)

        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, Tp, action_dim), device=self.device)
        naction = noisy_action
        
        # init scheduler
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = nets["policy"]["noise_pred_net"](
                sample=naction, 
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        # process action using Ta
        start = To - 1
        end = start + Ta
        action = naction[:,start:end]
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "optimizers": { k : self.optimizers[k].state_dict() for k in self.optimizers },
            "lr_schedulers": { k : self.lr_schedulers[k].state_dict() if self.lr_schedulers[k] is not None else None for k in self.lr_schedulers },
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict, load_optimizers=False):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
            load_optimizers (bool): whether to load optimizers and lr_schedulers from the model_dict;
                used when resuming training from a checkpoint
        """
        self.nets.load_state_dict(model_dict["nets"])

        # for backwards compatibility
        if "optimizers" not in model_dict:
            model_dict["optimizers"] = {}
        if "lr_schedulers" not in model_dict:
            model_dict["lr_schedulers"] = {}

        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])

        if load_optimizers:
            for k in model_dict["optimizers"]:
                self.optimizers[k].load_state_dict(model_dict["optimizers"][k])
            for k in model_dict["lr_schedulers"]:
                if model_dict["lr_schedulers"][k] is not None:
                    self.lr_schedulers[k].load_state_dict(model_dict["lr_schedulers"][k])


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version("1.9.0"):
        raise ImportError("This function requires pytorch >= 1.9.0")

    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split(".") for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module
