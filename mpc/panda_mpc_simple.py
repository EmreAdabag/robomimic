import torch
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
import time
import numpy as np


# Add paths
sys.path.insert(0, 'mpc.pytorch')
sys.path.insert(0, 'adam/src')
from mpc.mpc import MPC, QuadCost, GradMethods
from adam.pytorch.computation_batch import KinDynComputationsBatch
from adam.pytorch.torch_like import set_default_device

class PandaDynamics(torch.nn.Module):
    """GPU-batched Panda joint-space dynamics using ADAM (Torch+JAX) with fast vectorized FD Jacobians."""
    def __init__(self, urdf_path, dt=0.01, eps=1e-4, device=None):
        super().__init__()
        self.dt = dt
        self.eps = eps
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        # Ensure ADAM Torch backend creates tensors on the chosen device
        set_default_device(self.device)

        arm_joints = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                      'panda_joint5', 'panda_joint6', 'panda_joint7']
        # Build batched KinDyn with JAX JIT + vmap, exposed as torch funcs via jax2torch
        self.kin_b = KinDynComputationsBatch(urdf_path, joints_name_list=arm_joints)

        # Register joint limits as buffers so they follow device
        self.register_buffer(
            'q_lower',
            torch.tensor([-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671], dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            'q_upper',
            torch.tensor([2.9671, 1.8326, 2.9671, 0.0, 2.9671, 3.8223, 2.9671], dtype=torch.get_default_dtype())
        )

    def _compute_M_h(self, q, qd):
        """Compute joint-space mass matrix and bias for a batch.
        q, qd: [B,7]
        returns Mj: [B,7,7], hj: [B,7]
        """
        B = q.shape[0]
        H = torch.eye(4, dtype=q.dtype, device=q.device).expand(B, 4, 4)
        bv = torch.zeros(B, 6, dtype=q.dtype, device=q.device)
        M_full = self.kin_b.mass_matrix(H, q)               # [B, 13, 13] for 7 DoF + 6 base
        h_full = self.kin_b.bias_force(H, q, bv, qd)        # [B, 13]
        Mj = M_full[:, 6:, 6:]
        hj = h_full[:, 6:]
        # Regularize for numerical stability
        Mj = Mj + 1e-8 * torch.eye(7, dtype=q.dtype, device=q.device).unsqueeze(0)
        return Mj, hj

    def _step(self, x, u):
        """Discrete Euler step for joint dynamics; vectorized over batch."""
        q = x[:, :7]
        qd = x[:, 7:]
        q = torch.clamp(q, self.q_lower, self.q_upper)
        Mj, hj = self._compute_M_h(q, qd)
        # Solve qdd = M^{-1} (u - h)
        rhs = u - hj
        qdd = torch.linalg.solve(Mj, rhs.unsqueeze(-1)).squeeze(-1)
        qd_new = qd + self.dt * qdd
        q_new = q + self.dt * qd_new
        return torch.cat([q_new, qd_new], dim=-1)

    def forward(self, x, u):
        # x: [B,14], u: [B,7]
        return self._step(x, u)

    @torch.no_grad()
    def grad_input(self, x, u):
        """Vectorized forward-difference Jacobians on GPU.
        Returns R=df/dx in [B,14,14], S=df/du in [B,14,7].
        """
        B, nx = x.shape
        nu = u.shape[1]
        # Base evaluation
        f0 = self._step(x, u)  # [B,14]

        # State Jacobian R
        ex = self.eps
        # Build stacked perturbed states: for each sample i, perturb each state dim j independently
        x_rep = x.repeat_interleave(nx, dim=0)            # [B*nx, nx]
        u_rep = u.repeat_interleave(nx, dim=0)            # [B*nx, nu]
        # Add eps to the right indices
        eye_x = torch.eye(nx, dtype=x.dtype, device=x.device).repeat(B, 1)  # [B*nx, nx] in block-diagonal form via repeat
        x_pert = x_rep + ex * eye_x
        f_x = self._step(x_pert, u_rep)
        df_x = (f_x - f0.repeat_interleave(nx, dim=0)) / ex  # [B*nx, nx]
        R = df_x.view(B, nx, nx).transpose(1, 2)  # [B, nx_out, nx_in]

        # Control Jacobian S
        eu = self.eps
        x_rep2 = x.repeat_interleave(nu, dim=0)           # [B*nu, nx]
        u_rep2 = u.repeat_interleave(nu, dim=0)           # [B*nu, nu]
        eye_u = torch.eye(nu, dtype=u.dtype, device=u.device).repeat(B, 1)
        u_pert = u_rep2 + eu * eye_u
        f_u = self._step(x_rep2, u_pert)
        df_u = (f_u - f0.repeat_interleave(nu, dim=0)) / eu  # [B*nu, nx]
        S = df_u.view(B, nu, nx).transpose(1, 2)  # [B, nx_out, nu_in]

        return R, S

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sim_timestep = 0.005
    solve_timestep = 0.01

    # Initialize PyBullet
    # Use DIRECT to run in headless/any environment reliably
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(sim_timestep)
    
    # Load Panda URDF
    urdf_path = "/home/emrea/panda_pytorch/robot_description/panda_with_gripper.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)  # Fix base at ground level
    
    # Add ground plane
    p.loadURDF("plane.urdf", [0, 0, 0])
    
    # Set initial position and disable default motors
    initial_q = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5]
    for i in range(7):
        p.resetJointState(robot_id, i, initial_q[i])
        # Disable default motor control to allow torque control
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
    
    # Enable real-time simulation for better physics
    p.setRealTimeSimulation(0)  # Step simulation manually
    
    # Set camera view
    # p.resetDebugVisualizerCamera(
    #     cameraDistance=1.5,
    #     cameraYaw=45,
    #     cameraPitch=-30,
    #     cameraTargetPosition=[0, 0, 0.5]
    # )
    
    # MPC parameters
    n_state = 14  # 7 joint positions + 7 joint velocities
    n_ctrl = 7    # 7 joint torques
    T = 20        # Horizon length
    
    # Create GPU-batched dynamics model with vectorized FD Jacobians
    dynamics = PandaDynamics(urdf_path, dt=solve_timestep, device=device).to(device)
    
    # Create MPC controller
    mpc = MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        u_lower=None,  # Lower torque limits for stability
        u_upper=None,
        lqr_iter=10,
        verbose=0,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged=False,  # Don't crash on convergence issues
        eps=1e-6
    )
    
    # Target joint positions (safer home position)
    target_q =  torch.tensor([0.0, 0., 0.0, 0., 0.0, 0., 0.], device=device)
    target_state = torch.cat([target_q, torch.zeros(7, device=device)])  # Zero velocity target
    
    # Cost matrices - improved tuning
    Qdiag = torch.cat([1. * torch.ones(7), 0.01 * torch.ones(7), 0.00001 * torch.ones(7)])
    Q = torch.diag(Qdiag)
    Q = Q.repeat(T, 1, 1, 1)  # T x 1 x (n_state+n_ctrl) x (n_state+n_ctrl)
    cost = QuadCost(Q, torch.zeros((T, 1, n_state + n_ctrl)))
    
    # Initial state (current robot state)
    current_q = torch.tensor(initial_q, device=device)
    current_qdot = torch.zeros(7, device=device)
    x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)  # Add batch dimension
    
    print("Starting MPC control...")
    print(f"Target: {target_q.cpu().numpy()}")
    
    # time this mpc solve
    # MPC control loop
    for step in range(200):  # More steps for better convergence
        # Solve MPC problem

        
        # Apply first control action
        start_time = time.time()
        x_mpc, u_mpc, obj = mpc(x_init, cost, dynamics)
        end_time = time.time()
        print(f"MPC solve time: {end_time - start_time} seconds")
        # Use MPC-predicted next state's joint positions as position targets
        # Fallback to current target if horizon indexing is not available
        try:
            next_idx = 1 if x_mpc.shape[0] > 1 else 0
            q_cmd = x_mpc[next_idx, 0, :7].detach().cpu().numpy()
        except Exception:
            q_cmd = target_q.detach().cpu().numpy()

        # Apply position control in PyBullet
        for i in range(7):
            p.setJointMotorControl2(
                robot_id,
                i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(q_cmd[i])
            )
        
        # Step simulation
        for i in range(int(solve_timestep / sim_timestep)):
            p.stepSimulation()
            time.sleep(sim_timestep)
        
        # Get new state
        for i in range(7):
            current_q[i] = p.getJointState(robot_id, i)[0]
            current_qdot[i] = p.getJointState(robot_id, i)[1]
        
        x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)
        
        # Print progress and check convergence
        if step % 5 == 0:
            error = torch.norm(current_q - target_q)
            print(f"Step {step}, Error: {error:.4f}")
            print(f"  Current: {current_q.cpu().numpy()}")
            print(f"  Cmd Pos: {q_cmd}")
            
    
    print("Control completed!")
    p.disconnect()

if __name__ == "__main__":
    main()
