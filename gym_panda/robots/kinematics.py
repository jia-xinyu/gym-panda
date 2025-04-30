import quadprog
import numpy as np
import pinocchio as pin
import warnings

class pinKinematics:
    """Inverse kinematics sovler based on Quadprog.

    Args:
        model: Pinocchio model.
        data: Pinocchio data.
        tcp_id (int): Tip joint index.
        weight (np.ndarray, optional): Weight matrix of ee position and ee orientation.
        damping (float, optional): Damping coefficient of singularity protection.
    """
        
    def __init__(
        self,
        model,
        data,
        tcp_id: int,
        weight: np.ndarray = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        damping: float = 0.01,
    ) -> None:
        self.model = model
        self.data = data
        self.tcp_id = tcp_id
        self.weight = weight
        self.damping = damping


    def compute_fk(self, q: np.ndarray) -> pin.SE3:
        """Compute forward kinematics. 

        Args:
            q (np.ndarray): Joint angles.

        Returns:
            SE3: Tip frame placement.
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        return self.data.oMf[self.tcp_id]


    def compute_ik(self, Tdes: pin.SE3, q: np.ndarray, dt: float) -> np.ndarray:
        """Compute the inverse kinematics with joint constraints.
            
        min  1/2 x^T G x - a^T x
        s.t.   C.T x >= b

        Args:
            Tdes (pin.SE3): Target pose.
            q (np.ndarray): Current joint position.
            dt (float): Time step.
        Returns:
             np.ndarray: Joint velocity.
        """

        # Robot config
        dof = len(q)
        q_max = self.model.upperPositionLimit
        q_min = self.model.lowerPositionLimit
        v_limit = self.model.velocityLimit

        # Compute Jacobian
        T = self.compute_fk(q)
        V = pin.log( T.inverse() * Tdes ).vector / dt
        Jlocal = pin.computeFrameJacobian(self.model, self.data, q, self.tcp_id)

        # QP matrices for quadprog (0.5 x^T G x - a^T x)
        G = 2 * (Jlocal.T @ self.weight @ Jlocal + (self.damping ** 2) * np.eye(dof))
        # G = (G + G.T) / 2      # ensure symmetric

        f = -2 * (Jlocal.T @ self.weight @ V)
        a = -f

        # Constraints: Ax >= b
        A_ineq = np.vstack([np.eye(dof)*dt, -np.eye(dof)*dt])
        b_ineq = np.hstack([q_min-q, -(q_max-q)])

        # Bound constraints: [x, -x] >= [lb, -ub]
        lb = -v_limit
        ub = v_limit
        A_bound = np.vstack([np.eye(dof), -np.eye(dof)])
        b_bound = np.hstack([lb, -ub])

        # Combine constraints
        C = np.vstack([A_ineq, A_bound]).T  # quadprog wants C.T x >= b
        b_total = np.hstack([b_ineq, b_bound])

        # Solve the QP
        try:
            sol = quadprog.solve_qp(G, a, C, b_total)[0]
        except ValueError:
            sol = np.zeros(dof)  # fallback if solver fails
            warnings.warn("Failed to find solution. Using zeros as fallback.")

        return sol
