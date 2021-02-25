"""
Standard MPC for Passing through a dynamic gate
"""
import casadi as ca
import numpy as np
import time
from os import system

from neural_control.environments.copter import copter_params
from types import SimpleNamespace

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
copter_params = SimpleNamespace(**copter_params)
copter_params.translational_drag = torch.from_numpy(
    copter_params.translational_drag
).to(device)
copter_params.gravity = torch.from_numpy(copter_params.gravity).to(device)
copter_params.rotational_drag = torch.from_numpy(
    copter_params.rotational_drag
).to(device)
# estimate intertia as in flightmare
inertia_vector = (
    copter_params.mass / 12.0 * copter_params.arm_length**2 *
    torch.tensor([4.5, 4.5, 7])
).float().to(device)
copter_params.frame_inertia = torch.diag(inertia_vector)
# torch.from_numpy(copter_params.frame_inertia
#                                              ).float().to(device)
kinv_ang_vel_tau = torch.diag(torch.tensor([16.6, 16.6, 5.0]).float())


#
class MPC(object):
    """
    Nonlinear MPC
    """

    def __init__(self, T, dt, dynamics="high_mpc", so_path='./nmpc.so'):
        """
        Nonlinear MPC for quadrotor control        
        """
        self.so_path = so_path

        self.dynamics_model = dynamics

        # Time constant
        self._T = T
        self._dt = dt
        self._N = int(self._T / self._dt)

        # Gravity
        self._gz = 9.81

        if self.dynamics_model == "high_mpc":
            # Quadrotor constant
            self._w_max_yaw = 6.0
            self._w_min_yaw = -6.0
            self._w_max_xy = 6.0
            self._w_min_xy = -6.0
            self._thrust_min = 2.0
            self._thrust_max = 20.0
            # state dimension (px, py, pz,           # quadrotor position
            #                  qw, qx, qy, qz,       # quadrotor quaternion
            #                  vx, vy, vz,           # quadrotor linear velocity
            self._s_dim = 10
            # action dimensions (c_thrust, wx, wy, wz)
            self._u_dim = 4
        elif self.dynamics_model == "simple_quad":
            # Quadrotor constant
            self._w_max_yaw = 1
            self._w_min_yaw = 0
            self._w_max_xy = 1
            self._w_min_xy = 0
            self._thrust_min = 0
            self._thrust_max = 1
            self._s_dim = 12
            self._u_dim = 4
        elif self.dynamics_model == "fixed_wing":
            self._s_dim = 6
            self._u_dim = 2

        # cost matrix for tracking the goal point
        self._Q_goal = np.diag(
            [
                100,
                100,
                100,  # delta_x, delta_y, delta_z
                0,
                0,
                0,
                0,  # delta_qw, delta_qx, delta_qy, delta_qz
                10,
                10,
                10
            ]
        )

        # cost matrix for tracking the pendulum motion
        self._Q_pen = np.diag(
            [
                0,
                100,
                100,  # delta_x, delta_y, delta_z
                0,
                0,
                0,
                0,  # delta_qw, delta_qx, delta_qy, delta_qz
                0,
                10,
                10
            ]
        )  # delta_vx, delta_vy, delta_vz

        # cost matrix for the action
        self._Q_u = np.diag([0.1, 0.1, 0.1, 0.1])  # T, wx, wy, wz

        # initial state and control action
        self._quad_s0 = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._quad_u0 = [9.81, 0.0, 0.0, 0.0]

        self._initDynamics()

    def _initDynamics(self, ):
        # # # # # # # # # # # # # # # # # # #
        # ---------- Input States -----------
        # # # # # # # # # # # # # # # # # # #

        # # Fold
        if self.dynamics_model == "high_mpc":
            F = self.drone_dynamics_high_mpc(self._dt)
        elif self.dynamics_model == "simple_quad":
            F = self.drone_dynamics_simple(self.dt)
        fMap = F.map(self._N, "openmp")  # parallel

        # # # # # # # # # # # # # # #
        # ---- loss function --------
        # # # # # # # # # # # # # # #

        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self._s_dim)
        Delta_p = ca.SX.sym("Delta_p", self._s_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        #
        cost_goal = Delta_s.T @ self._Q_goal @ Delta_s
        cost_gap = Delta_p.T @ self._Q_pen @ Delta_p
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        #
        f_cost_goal = ca.Function('cost_goal', [Delta_s], [cost_goal])
        f_cost_gap = ca.Function('cost_gap', [Delta_p], [cost_gap])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        #
        # # # # # # # # # # # # # # # # # # # #
        # # ---- Non-linear Optimization -----
        # # # # # # # # # # # # # # # # # # # #
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw
        #
        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [
            self._thrust_min, self._w_min_xy, self._w_min_xy, self._w_min_yaw
        ]
        u_max = [
            self._thrust_max, self._w_max_xy, self._w_max_xy, self._w_max_yaw
        ]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self._s_dim)]
        x_max = [+x_bound for _ in range(self._s_dim)]
        #
        g_min = [0 for _ in range(self._s_dim)]
        g_max = [0 for _ in range(self._s_dim)]

        P = ca.SX.sym(
            "P", self._s_dim + (self._s_dim + 3) * self._N + self._s_dim
        )
        X = ca.SX.sym("X", self._s_dim, self._N + 1)
        U = ca.SX.sym("U", self._u_dim, self._N)
        #
        X_next = fMap(X[:, :self._N], U)

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._quad_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0:self._s_dim]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self._N):
            #
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._quad_u0
            self.lbw += u_min
            self.ubw += u_max

            # retrieve time constant
            # idx_k = self._s_dim+self._s_dim+(self._s_dim+3)*(k)
            # idx_k_end = self._s_dim+(self._s_dim+3)*(k+1)
            # time_k = P[ idx_k : idx_k_end]

            # cost for tracking the goal position
            cost_goal_k, cost_gap_k = 0, 0
            if k >= self._N - 1:  # The goal postion.
                delta_s_k = (
                    X[:, k + 1] - P[self._s_dim + (self._s_dim + 3) * self._N:]
                )
                cost_goal_k = f_cost_goal(delta_s_k)
            else:
                # cost for tracking the moving gap
                delta_p_k = (X[:, k+1] - P[self._s_dim+(self._s_dim+3)*k : \
                    self._s_dim+(self._s_dim+3)*(k+1)-3])
                cost_gap_k = f_cost_gap(delta_p_k)

            delta_u_k = U[:, k] - [self._gz, 0, 0, 0]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_goal_k + cost_u_k + cost_gap_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w0 += self._quad_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {
            'f': self.mpc_obj,
            'x': ca.vertcat(*self.nlp_w),
            'p': P,
            'g': ca.vertcat(*self.nlp_g)
        }

        # # # # # # # # # # # # # # # # # # #
        # -- ipopt
        # # # # # # # # # # # # # # # # # # #
        ipopt_options = {
            'verbose': False, \
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):
        # # # # # # # # # # # # # # # #
        # -------- solve NLP ---------
        # # # # # # # # # # # # # # # #
        #
        if self.dynamics_model == "high_mpc":
            start = ref_states[:self._s_dim]
            end = ref_states[-self._s_dim:]
            end[3:7] = [0 for _ in range(4)]
            middle_ref_states = np.array(ref_states[self._s_dim:-self._s_dim]
                                         ).reshape((10, 13))
            middle_ref_states[:, 3:7] = 0
            ref_states = start + middle_ref_states.flatten().tolist() + end

        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            p=ref_states,
            lbg=self.lbg,
            ubg=self.ubg
        )
        #
        sol_x0 = self.sol['x'].full()
        opt_u = sol_x0[self._s_dim:self._s_dim + self._u_dim]

        # Warm initialization
        self.nlp_w0 = list(
            sol_x0[self._s_dim + self._u_dim:2 * (self._s_dim + self._u_dim)]
        ) + list(sol_x0[self._s_dim + self._u_dim:])
        # print(self.nlp_w0)
        # print(len(self.nlp_w0))
        # #
        x0_array = np.reshape(
            sol_x0[:-self._s_dim], newshape=(-1, self._s_dim + self._u_dim)
        )
        # print(len(sol_x0))
        # for i in range(10):
        #     traj_test = sol_x0[i*14 : (i+1)*14]
        #     print([round(s[0],2) for s in traj_test])
        print(opt_u)
        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array

    def drone_dynamics_high_mpc(self, dt):

        self.f = self.get_dynamics_high_mpc()

        M = 4  # refinement
        DT = dt / M
        X0 = ca.SX.sym("X", self._s_dim)
        U = ca.SX.sym("U", self._u_dim)
        # #
        X = X0
        for _ in range(M):
            # --------- RK4------------
            k1 = DT * self.f(X, U)
            k2 = DT * self.f(X + 0.5 * k1, U)
            k3 = DT * self.f(X + 0.5 * k2, U)
            k4 = DT * self.f(X + k3, U)
            #
            X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # Fold
        F = ca.Function('F', [X0, U], [X])
        return F

    def get_dynamics_high_mpc(self):
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        #
        qw, qx, qy, qz = ca.SX.sym('qw'), ca.SX.sym('qx'), ca.SX.sym('qy'), \
            ca.SX.sym('qz')
        #
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), \
            ca.SX.sym('wy'), ca.SX.sym('wz')

        # -- conctenated vector
        self._u = ca.vertcat(thrust, wx, wy, wz)

        # # # # # # # # # # # # # # # # # # #
        # --------- System Dynamics ---------
        # # # # # # # # # # # # # # # # # # #

        x_dot = ca.vertcat(
            vx, vy, vz, 0.5 * (-wx * qx - wy * qy - wz * qz),
            0.5 * (wx * qw + wz * qy - wy * qz),
            0.5 * (wy * qw - wz * qx + wx * qz),
            0.5 * (wz * qw + wy * qx - wx * qy),
            2 * (qw * qy + qx * qz) * thrust, 2 * (qy * qz - qw * qx) * thrust,
            (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self._gz
            # (1 - 2*qx*qx - 2*qy*qy) * thrust - self._gz
        )

        #
        func = ca.Function(
            'f', [self._x, self._u], [x_dot], ['x', 'u'], ['ode']
        )
        return func

    def drone_dynamics_simple(self, dt):

        # # # # # # # # # # # # # # # # # # #
        # --------- State ------------
        # # # # # # # # # # # # # # # # # # #

        # position
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        # attitude
        ax, ay, az = ca.SX.sym('ax'), ca.SX.sym('ay'), ca.SX.sym('az')
        # vel
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # angular velocity
        avx, avy, avz = ca.SX.sym('avx'), ca.SX.sym('avy'), ca.SX.sym('avz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, ax, ay, az, vx, vy, vz, avx, avy, avz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), \
            ca.SX.sym('wy'), ca.SX.sym('wz')

        # -- conctenated vector
        self._u = ca.vertcat(thrust, wx, wy, wz)

        thrust_scaled = thrust * 10 - 5 + 7
        body_rates = ca.vertcat(wx - .5, wy - .5, wz - .5)

        # linear dynamics
        Cy = ca.cos(az)
        Sy = ca.sin(az)
        Cp = ca.cos(ay)
        Sp = ca.sin(ay)
        Cr = ca.cos(ax)
        Sr = ca.sin(ax)

        const = thrust_scaled / copter_params.mass
        acc_x = (Cy * Sp * Cr + Sr * Sy) * const
        acc_y = (Cr * Sy * Sp - Cy * Sr) * const
        acc_z = (Cr * Cp) * const - 9.81

        px_new = px + 0.5 * dt * dt * acc_x + 0.5 * dt * vx
        py_new = py + 0.5 * dt * dt * acc_y + 0.5 * dt * vy
        pz_new = pz + 0.5 * dt * dt * acc_z + 0.5 * dt * vz
        vx_new = vx + dt * acc_x
        vy_new = vy + dt * acc_y
        vz_new = vz + dt * acc_z

        # angular dynamics

        att_new = 0
        av_new = 0

        # stack together
        X = ca.vertcat(
            px_new, py_new, pz_new, att_new, vx_new, vy_new, vz_new, av_new
        )
        # Fold
        F = ca.Function('F', [self._x, self._u], [X])
        return F
