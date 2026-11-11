# Theory: Robust Control and Energy-Efficient Trajectory Planning of a 3R Planar Arm

This document explains every model and algorithm used in the code, in the order
you need them: geometry first, then motion, then forces, then planning and control.
Each derivation ends with a short **"In plain English"** note saying what the result means.

| Part | Topic | Code |
|---|---|---|
| 1 | Robot description and notation | `robot3r/robot.py` |
| 2 | Forward kinematics | `RobotArm3R.forward_kinematics` |
| 3 | Jacobian and singularities | `RobotArm3R.jacobian`, `manipulability` |
| 4 | Inverse kinematics of a redundant arm | `RobotArm3R.inverse_kinematics`, `ik_numerical` |
| 5 | Dynamic model (Lagrange) with friction | `mass_matrix`, `coriolis_matrix`, `gravity_vector`, `friction_torque` |
| 6 | Trajectory planning: quintic and optimal | `robot3r/trajectory.py`, `robot3r/optimization.py` |
| 7 | Controllers: PID, CTC, SMC | `robot3r/controllers.py` |
| 8 | Uncertainty, disturbances, metrics | `robot3r/disturbances.py`, `robot3r/simulation.py` |

---

## Part 1 — Robot description and notation

### 1.1 Physical parameters

| Quantity | Link 1 | Link 2 | Link 3 |
|---|---|---|---|
| Length $l_i$ [m] | 1.0 | 0.8 | 0.6 |
| Mass $m_i$ [kg] | 1.0 | 0.8 | 0.5 |
| COM distance $l_{ci} = l_i/2$ [m] | 0.5 | 0.4 | 0.3 |
| Inertia about COM $I_i = \tfrac{1}{12}m_il_i^2$ [kg·m²] | 0.08333 | 0.04267 | 0.01500 |
| Viscous friction $b_i$ [N·m·s/rad] | 0.25 | 0.20 | 0.15 |
| Coulomb friction $f_{ci}$ [N·m] (optional) | 0.30 | 0.20 | 0.10 |

Gravity is $g = 9.81$ m/s² and acts in the $-y$ direction.

### 1.2 Angles and shorthand

The joint vector is $\boldsymbol\theta = [\theta_1,\ \theta_2,\ \theta_3]^\top$.

* $\theta_1$ is measured from the world $x$-axis.
* $\theta_2$ and $\theta_3$ are **relative** angles: each is measured from the previous link.
* Positive angles and torques are counter-clockwise.

The absolute angle of each link is a running sum, so we use this shorthand:

$$
\theta_{12} = \theta_1 + \theta_2,\qquad \theta_{123} = \theta_1 + \theta_2 + \theta_3,
$$

$$
c_1 = \cos\theta_1,\quad c_{12} = \cos\theta_{12},\quad c_{123} = \cos\theta_{123},\quad
c_{23} = \cos(\theta_2 + \theta_3),\quad \text{(and } s_\ast \text{ for sines).}
$$

---

## Part 2 — Forward kinematics

### 2.1 Step 1: one transform per joint

Each joint rotates about $z$ and is followed by a translation of length $l_i$ along the new $x$-axis.
In homogeneous coordinates (standard DH with $a_i = l_i$, $\alpha_i = d_i = 0$):

$$
{}^{i-1}T_i =
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i & 0 & l_i\cos\theta_i\\
\sin\theta_i & \cos\theta_i & 0 & l_i\sin\theta_i\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

### 2.2 Step 2: chain the transforms

Because all axes are parallel, multiplying planar rotations simply **adds the angles**:

$$
{}^{0}T_3 = {}^{0}T_1\,{}^{1}T_2\,{}^{2}T_3 =
\begin{bmatrix}
c_{123} & -s_{123} & 0 & x\\
s_{123} & c_{123} & 0 & y\\
0&0&1&0\\0&0&0&1
\end{bmatrix}.
$$

### 2.3 Step 3: read off the pose

$$
\boxed{
\begin{aligned}
x &= l_1c_1 + l_2c_{12} + l_3c_{123}\\
y &= l_1s_1 + l_2s_{12} + l_3s_{123}\\
\phi &= \theta_1 + \theta_2 + \theta_3
\end{aligned}}
$$

The intermediate joint positions, used for drawing and for collision constraints, are

$$
\mathbf o_1 = l_1\begin{bmatrix}c_1\\ s_1\end{bmatrix},\qquad
\mathbf o_2 = \mathbf o_1 + l_2\begin{bmatrix}c_{12}\\ s_{12}\end{bmatrix},\qquad
\mathbf p = \mathbf o_2 + l_3\begin{bmatrix}c_{123}\\ s_{123}\end{bmatrix}.
$$

> **In plain English:** the end-effector position is the sum of three arrows, one per link.
> Each arrow points in the direction of that link's *absolute* angle. The tool orientation $\phi$ is
> just the sum of all joint angles.

**Workspace.** The reachable positions form an annulus with outer radius $l_1 + l_2 + l_3 = 2.4$ m.
Because $l_1 < l_2 + l_3$, the inner radius is $0$: every point within 2.4 m of the base is reachable.

---

## Part 3 — Jacobian and singularities

### 3.1 Geometric Jacobian

For a revolute joint $i$ with axis $\mathbf z$ located at $\mathbf o_{i-1}$, the Jacobian column is
$\mathbf J_i = \big[\,(\mathbf z \times (\mathbf p - \mathbf o_{i-1}))^\top,\ \mathbf z^\top\big]^\top$.
In the plane, $\mathbf z \times [a, b, 0]^\top = [-b, a, 0]^\top$. Keeping the three non-trivial rows
$(\dot x, \dot y, \dot\phi)$:

$$
\boxed{
\mathbf J(\boldsymbol\theta) =
\begin{bmatrix}
-l_1s_1 - l_2s_{12} - l_3s_{123} & -l_2s_{12} - l_3s_{123} & -l_3s_{123}\\
\ \ \,l_1c_1 + l_2c_{12} + l_3c_{123} & \ \ \,l_2c_{12} + l_3c_{123} & \ \ \,l_3c_{123}\\
1 & 1 & 1
\end{bmatrix}},
\qquad
\begin{bmatrix}\dot x\\ \dot y\\ \dot\phi\end{bmatrix} = \mathbf J(\boldsymbol\theta)\,\dot{\boldsymbol\theta}.
$$

The top two rows form the **positional Jacobian** $\mathbf J_p \in \mathbb R^{2\times3}$, which is the one that
matters when only the end-effector *position* is specified. The code also provides the full $6\times3$ spatial form,
where the extra rows are zero.

> **In plain English:** column $i$ says how the tool moves if only joint $i$ turns. It turns about its own
> axis, so the tool moves perpendicular to the line from that joint to the tool, at a speed proportional to the
> distance.

### 3.2 Singularities of the full (position + orientation) task

Subtracting columns (which does not change the determinant) gives

$$
\det\mathbf J = l_1l_2\sin\theta_2 .
$$

The full pose task is singular when $\theta_2 = 0$ or $\theta_2 = \pi$, i.e. when links 1 and 2 are aligned.
Joint 3 does not appear, because it only rotates the last link about the wrist.

### 3.3 Singularities of the position-only task

$\mathbf J_p$ is $2\times3$, so it loses rank only if **all three** columns become parallel. By the Cauchy–Binet
formula, the manipulability measure is

$$
w(\boldsymbol\theta) = \sqrt{\det(\mathbf J_p\mathbf J_p^\top)}
= \sqrt{(l_1l_2s_2 + l_1l_3s_{23})^2 + (l_1l_3s_{23} + l_2l_3s_3)^2 + (l_2l_3s_3)^2}.
$$

$w = 0$ requires $s_3 = 0$, then $s_{23} = 0$, then $s_2 = 0$. In words, **all links must be collinear**
(fully stretched or folded back).

> **In plain English:** a singularity is a posture where the arm loses the ability to move the tool in some
> direction. Because the 3R arm has one joint more than a planar position task needs, it is singular for
> positioning only when it is completely straight or completely folded, which is much rarer than for a 2R arm.
> The number $w$ measures how far the arm is from such a posture.

### 3.4 How the code handles singularities

* `singularity_index(q)` returns the smallest singular value of $\mathbf J_p$ (or of $\mathbf J$ for the pose task),
  and `is_singular(q)` compares it with a tolerance.
* All Jacobian inverses use **damped least squares (DLS)**:

$$
\mathbf J^{\#} = \mathbf J^\top\big(\mathbf J\mathbf J^\top + \lambda^2\mathbf I\big)^{-1},
\qquad
\lambda^2 =
\begin{cases}
0, & \sigma_{\min} \ge \varepsilon\\
\lambda_{\max}^2\big(1 - (\sigma_{\min}/\varepsilon)^2\big), & \sigma_{\min} < \varepsilon
\end{cases}
$$

  Damping is zero far from singularities, so the exact pseudo-inverse is used there. It grows smoothly as the
  smallest singular value $\sigma_{\min}$ approaches zero, which keeps joint velocities bounded.
* The IK redundancy resolution (Part 4) **maximises $w$**, so planned postures stay away from singularities.

---

## Part 4 — Inverse kinematics of a redundant arm

### 4.1 Why the 3R arm is redundant

For a position-only task we must satisfy **2 equations** ($x$ and $y$) with **3 unknowns**. For almost every
reachable point there is therefore a **one-parameter family** of solutions, called the *self-motion manifold*:
the arm can move its joints without moving the tool.

To get a unique answer we must add one condition. Two classical choices exist:

| Method | Idea | Pros | Cons |
|---|---|---|---|
| **Geometric + configuration selection** (main method) | Parameterise the family by the tool angle $\phi$, solve in closed form, choose $\phi$ by optimising a secondary criterion | Exact, fast, enumerates all solutions, handles elbow branches | Needs a 1-D search over $\phi$ |
| **Numerical (differential IK)** (verification and alternative) | Iterate $\Delta\boldsymbol\theta = \mathbf J_p^{\#}\mathbf e + (\mathbf I - \mathbf J_p^{\#}\mathbf J_p)\,k\nabla H$ | General, works for any arm | Local, depends on the initial guess, may converge to different branches |

We use the **geometric method** as the primary IK. It is exact, and it makes the redundancy explicit and controllable.

### 4.2 Step 1: fix the tool angle and find the wrist

Pick a tool angle $\phi$. The wrist (joint 3) must then sit one link length behind the tool:

$$
\mathbf p_w = \begin{bmatrix}x_w\\ y_w\end{bmatrix} = \begin{bmatrix}x - l_3\cos\phi\\ y - l_3\sin\phi\end{bmatrix}.
$$

### 4.3 Step 2: solve the remaining 2R problem

Links 1 and 2 must reach $\mathbf p_w$. This is the classical 2R problem (law of cosines):

$$
c_2 = \frac{x_w^2 + y_w^2 - l_1^2 - l_2^2}{2l_1l_2},\qquad
\theta_2 = \operatorname{atan2}\!\big(\sigma\sqrt{1 - c_2^2},\ c_2\big),\qquad \sigma = \pm1,
$$

$$
\theta_1 = \operatorname{atan2}(y_w,\ x_w) - \operatorname{atan2}\!\big(l_2\sin\theta_2,\ l_1 + l_2\cos\theta_2\big).
$$

$\sigma = -1$ is the **elbow-up** branch ($\theta_2 < 0$) and $\sigma = +1$ the **elbow-down** branch.

### 4.4 Step 3: close the orientation

$$
\theta_3 = \phi - \theta_1 - \theta_2 .
$$

### 4.5 Which tool angles are allowed?

Step 2 only has a solution if the wrist is reachable by links 1–2, i.e.

$$
|l_1 - l_2| \;\le\; \|\mathbf p_w(\phi)\| \;\le\; l_1 + l_2 .
$$

As $\phi$ sweeps a full circle, the wrist traces a circle of radius $l_3$ around the target. The admissible
angles $\Phi_{\text{feas}}(\mathbf p)$ are the arcs of that circle lying inside the 2R annulus. The code samples
$\phi$ on a fine grid, keeps the admissible samples, and also rejects any that violate the joint limits.

> **In plain English:** imagine holding the tool tip fixed and rotating the last link around it. The wrist
> slides along a small circle. Wherever that circle is reachable by the first two links, we get a valid arm
> posture, actually two (elbow up or down). The free angle $\phi$ is exactly the redundancy.

### 4.6 Step 4: choose the best configuration

Among all admissible $\phi$ (and the chosen elbow branch) we pick

$$
\phi^\ast = \arg\max_{\phi \in \Phi_{\text{feas}}}\ \Big[\, w\big(\boldsymbol\theta(\phi)\big)
\;-\; k_c\,\big\|\boldsymbol\theta(\phi) - \boldsymbol\theta_{\text{prev}}\big\|^2
\;-\; k_l\,\textstyle\sum_i \rho_i\big(\theta_i(\phi)\big)\Big].
$$

* $w$ is the manipulability of §3.3, which keeps the arm far from singularities.
* The **continuity term** (weight $k_c$, optional) prefers postures close to a previous one, so a sequence of via
  points produces a smooth joint path without branch flips.
* $\rho_i$ is a smooth penalty that grows near the joint limits.

The search is a dense grid over $\phi$ (0.5° spacing) followed by a bounded Brent refinement
(`scipy.optimize.minimize_scalar`) around the best grid point. If the user specifies $\phi$, Steps 1–3 give the
answer directly and no search is needed: the arm is then not redundant.

### 4.7 Numerical alternative: differential IK with null-space optimisation

Starting from a guess $\boldsymbol\theta_0$, iterate

$$
\boldsymbol\theta_{k+1} = \boldsymbol\theta_k
+ \underbrace{\mathbf J_p^{\#}\,(\mathbf p_d - \mathbf p(\boldsymbol\theta_k))}_{\text{reduce the position error}}
+ \underbrace{\big(\mathbf I - \mathbf J_p^{\#}\mathbf J_p\big)\,k_H\,\nabla_{\boldsymbol\theta} w}_{\text{improve manipulability without moving the tool}} .
$$

The projector $\mathbf I - \mathbf J_p^{\#}\mathbf J_p$ maps any joint motion into the **null space** of $\mathbf J_p$.
Motions there do not move the end-effector, so the secondary objective can be improved for free.
The code uses this method to cross-check the geometric IK.

> **In plain English:** the numerical method walks downhill on the position error. Each step it also takes a
> sideways step along the self-motion manifold to make the posture more dexterous.

---

## Part 5 — Dynamic model (Lagrangian mechanics with friction)

### 5.1 Step 1: positions of the centres of mass

$$
\mathbf p_{c1} = l_{c1}\begin{bmatrix}c_1\\ s_1\end{bmatrix},\quad
\mathbf p_{c2} = \mathbf o_1 + l_{c2}\begin{bmatrix}c_{12}\\ s_{12}\end{bmatrix},\quad
\mathbf p_{c3} = \mathbf o_2 + l_{c3}\begin{bmatrix}c_{123}\\ s_{123}\end{bmatrix}.
$$

### 5.2 Step 2: velocities

Define the absolute angular rates $\omega_1 = \dot\theta_1$, $\omega_2 = \dot\theta_1 + \dot\theta_2$ and
$\omega_3 = \dot\theta_1 + \dot\theta_2 + \dot\theta_3$. Differentiating, and using $\cos(a - b) = c_ac_b + s_as_b$:

$$
\begin{aligned}
\|\dot{\mathbf p}_{c1}\|^2 &= l_{c1}^2\,\omega_1^2\\[2pt]
\|\dot{\mathbf p}_{c2}\|^2 &= l_1^2\omega_1^2 + l_{c2}^2\omega_2^2 + 2l_1l_{c2}\,c_2\,\omega_1\omega_2\\[2pt]
\|\dot{\mathbf p}_{c3}\|^2 &= l_1^2\omega_1^2 + l_2^2\omega_2^2 + l_{c3}^2\omega_3^2
+ 2l_1l_2\,c_2\,\omega_1\omega_2 + 2l_1l_{c3}\,c_{23}\,\omega_1\omega_3 + 2l_2l_{c3}\,c_3\,\omega_2\omega_3
\end{aligned}
$$

> **In plain English:** each COM velocity is a sum of rotating vectors. The cross terms depend only on the
> *relative* angles between links ($\theta_2$, $\theta_3$, $\theta_2 + \theta_3$), never on $\theta_1$.

### 5.3 Step 3: kinetic and potential energy

$$
T = \sum_{i=1}^{3}\Big(\tfrac12 m_i\|\dot{\mathbf p}_{ci}\|^2 + \tfrac12 I_i\,\omega_i^2\Big)
= \tfrac12\,\dot{\boldsymbol\theta}^\top\mathbf M(\boldsymbol\theta)\,\dot{\boldsymbol\theta},
$$

$$
V = g\big[m_1l_{c1}s_1 + m_2(l_1s_1 + l_{c2}s_{12}) + m_3(l_1s_1 + l_2s_{12} + l_{c3}s_{123})\big].
$$

### 5.4 Step 4: Lagrange's equations with friction

Friction does not come from a potential, so it enters as a generalised non-conservative force:

$$
\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot\theta_i} - \frac{\partial\mathcal L}{\partial\theta_i}
= \tau_i - F_i(\dot\theta_i), \qquad \mathcal L = T - V .
$$

Viscous friction can equivalently be written with the Rayleigh dissipation function
$\mathcal D = \tfrac12\dot{\boldsymbol\theta}^\top\mathbf B\dot{\boldsymbol\theta}$, giving $F = \partial\mathcal D/\partial\dot{\boldsymbol\theta} = \mathbf B\dot{\boldsymbol\theta}$.

Since $T$ is quadratic in $\dot{\boldsymbol\theta}$, these equations always take the standard form

$$
\boxed{\ \mathbf M(\boldsymbol\theta)\ddot{\boldsymbol\theta} + \mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + \mathbf G(\boldsymbol\theta) + \mathbf F(\dot{\boldsymbol\theta}) = \boldsymbol\tau\ }
$$

The next sections give each term explicitly.

### 5.5 The inertia matrix $\mathbf M(\boldsymbol\theta)$

Collecting the coefficients of $\dot\theta_i\dot\theta_j$ in $T$, we introduce **six constants**:

$$
\begin{aligned}
a_1 &= I_1 + I_2 + I_3 + m_1l_{c1}^2 + m_2(l_1^2 + l_{c2}^2) + m_3(l_1^2 + l_2^2 + l_{c3}^2)\\
a_2 &= I_2 + I_3 + m_2l_{c2}^2 + m_3(l_2^2 + l_{c3}^2)\\
a_3 &= I_3 + m_3l_{c3}^2\\
b_1 &= l_1(m_2l_{c2} + m_3l_2),\qquad b_2 = m_3l_1l_{c3},\qquad b_3 = m_3l_2l_{c3}
\end{aligned}
$$

Then

$$
\boxed{
\mathbf M(\boldsymbol\theta) =
\begin{bmatrix}
a_1 + 2b_1c_2 + 2b_2c_{23} + 2b_3c_3 & a_2 + b_1c_2 + b_2c_{23} + 2b_3c_3 & a_3 + b_2c_{23} + b_3c_3\\
\cdot & a_2 + 2b_3c_3 & a_3 + b_3c_3\\
\cdot & \cdot & a_3
\end{bmatrix}}
\quad(\text{symmetric}).
$$

**Numerical values for this robot:**

| $a_1$ | $a_2$ | $a_3$ | $b_1$ | $b_2$ | $b_3$ |
|---|---|---|---|---|---|
| 2.18400 | 0.55067 | 0.06000 | 0.72000 | 0.15000 | 0.12000 |

> **In plain English:** $\mathbf M$ says how hard it is to accelerate each joint. $M_{11}$ is the inertia felt
> by the shoulder: it is largest when the arm is stretched (all cosines $= 1$) and smallest when folded.
> Here it ranges over roughly $[0.7,\ 4.2]$ kg·m², a factor of about six. Off-diagonal terms mean that
> accelerating one joint pushes on the others. $\mathbf M$ never depends on $\theta_1$, because rotating the
> whole arm does not change its shape. It is always symmetric and positive definite, so it can always be inverted.

### 5.6 The Coriolis and centrifugal matrix $\mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta})$

We build $\mathbf C$ from the **Christoffel symbols of the first kind**:

$$
C_{ij} = \sum_{k=1}^{3} c_{ijk}\,\dot\theta_k,
\qquad
c_{ijk} = \frac12\left(\frac{\partial M_{ij}}{\partial\theta_k} + \frac{\partial M_{ik}}{\partial\theta_j} - \frac{\partial M_{jk}}{\partial\theta_i}\right).
$$

Only derivatives with respect to $\theta_2$ and $\theta_3$ are non-zero. Introduce the three
configuration-dependent coefficients

$$
\beta_1 = b_1s_2,\qquad \beta_2 = b_2s_{23},\qquad \beta_3 = b_3s_3,
$$

and write $\dot\theta_{12} = \dot\theta_1 + \dot\theta_2$ and $\dot\theta_{123} = \dot\theta_1 + \dot\theta_2 + \dot\theta_3$. Carrying out the sums gives

$$
\boxed{
\mathbf C =
\begin{bmatrix}
-\beta_1\dot\theta_2 - \beta_2(\dot\theta_2 + \dot\theta_3) - \beta_3\dot\theta_3 &
-(\beta_1 + \beta_2)\,\dot\theta_{12} - (\beta_2 + \beta_3)\,\dot\theta_3 &
-(\beta_2 + \beta_3)\,\dot\theta_{123}\\[4pt]
(\beta_1 + \beta_2)\,\dot\theta_1 - \beta_3\dot\theta_3 &
-\beta_3\dot\theta_3 &
-\beta_3\,\dot\theta_{123}\\[4pt]
(\beta_2 + \beta_3)\,\dot\theta_1 + \beta_3\dot\theta_2 &
\beta_3\,\dot\theta_{12} &
0
\end{bmatrix}}
$$

Multiplying out, the velocity-dependent torques $\mathbf h = \mathbf C\dot{\boldsymbol\theta}$ are

$$
\begin{aligned}
h_1 &= -\beta_1\,\dot\theta_2(2\dot\theta_1 + \dot\theta_2) \;-\; \beta_2\,(\dot\theta_2 + \dot\theta_3)(2\dot\theta_1 + \dot\theta_2 + \dot\theta_3) \;-\; \beta_3\,\dot\theta_3(2\dot\theta_1 + 2\dot\theta_2 + \dot\theta_3)\\
h_2 &= (\beta_1 + \beta_2)\,\dot\theta_1^2 \;-\; \beta_3\,\dot\theta_3(2\dot\theta_1 + 2\dot\theta_2 + \dot\theta_3)\\
h_3 &= \beta_2\,\dot\theta_1^2 + \beta_3\,(\dot\theta_1 + \dot\theta_2)^2
\end{aligned}
$$

**Key structural property (passivity).** With the Christoffel choice, $\dot{\mathbf M} - 2\mathbf C$ is
skew-symmetric, so $\mathbf x^\top(\dot{\mathbf M} - 2\mathbf C)\mathbf x = 0$ for every $\mathbf x$. This property is
used in the sliding-mode stability proof (§7.3), and the test suite checks it numerically.

> **In plain English:** these are the "fictitious" torques that appear because the links rotate. Squared
> velocities ($\dot\theta_1^2$) are **centrifugal** effects: link 3 is flung outward when link 1 spins, which is
> exactly $h_3$. Products of velocities ($\dot\theta_1\dot\theta_2$) are **Coriolis** effects. They grow with
> speed squared, so they are negligible for slow motions and important for fast ones.

### 5.7 The gravity vector $\mathbf G(\boldsymbol\theta)$

$\mathbf G = \partial V/\partial\boldsymbol\theta$. With

$$
\gamma_1 = g\,\big(m_1l_{c1} + (m_2 + m_3)\,l_1\big),\qquad
\gamma_2 = g\,(m_2l_{c2} + m_3l_2),\qquad
\gamma_3 = g\,m_3l_{c3},
$$

$$
\boxed{
\mathbf G(\boldsymbol\theta) =
\begin{bmatrix}
\gamma_1c_1 + \gamma_2c_{12} + \gamma_3c_{123}\\
\gamma_2c_{12} + \gamma_3c_{123}\\
\gamma_3c_{123}
\end{bmatrix}},
\qquad
\gamma_1 = 17.658,\ \ \gamma_2 = 7.063,\ \ \gamma_3 = 1.472\ \ [\text{N·m}].
$$

> **In plain English:** joint $i$ must hold up everything outboard of it. Each term is a weight times its
> horizontal lever arm. The shoulder carries the most: up to $26.2$ N·m with the arm horizontal and straight.
> With a 1 kg payload this nearly doubles.

### 5.8 The friction term $\mathbf F(\dot{\boldsymbol\theta})$

$$
\boxed{\ \mathbf F(\dot{\boldsymbol\theta}) = \mathbf B\,\dot{\boldsymbol\theta} + \mathbf F_c\tanh\!\big(\dot{\boldsymbol\theta}/\varepsilon\big)\ },
\qquad \mathbf B = \operatorname{diag}(0.25,\ 0.20,\ 0.15),\quad \mathbf F_c = \operatorname{diag}(0.30,\ 0.20,\ 0.10).
$$

The viscous term is the required model. The Coulomb term is optional (`coulomb_friction=(0,0,0)` disables it) and is
smoothed with $\tanh(\cdot/\varepsilon)$, $\varepsilon = 0.05$ rad/s. Smoothing avoids the discontinuity of
$\operatorname{sign}(\dot\theta)$, which would make the ODE non-smooth and the integrator inefficient.

> **In plain English:** friction always opposes motion. The viscous part grows with speed. The Coulomb part
> is a roughly constant drag that switches direction when the joint reverses.

### 5.9 Final equations of motion and state-space form

$$
\boxed{\ \mathbf M(\boldsymbol\theta)\ddot{\boldsymbol\theta} + \mathbf C(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + \mathbf G(\boldsymbol\theta) + \mathbf F(\dot{\boldsymbol\theta}) = \boldsymbol\tau + \boldsymbol\tau_d\ }
$$

$\boldsymbol\tau_d$ is an external disturbance torque (Part 8). For simulation, with
$\mathbf x = [\boldsymbol\theta;\ \dot{\boldsymbol\theta}]$:

$$
\dot{\mathbf x} =
\begin{bmatrix}
\dot{\boldsymbol\theta}\\
\mathbf M^{-1}\big(\boldsymbol\tau + \boldsymbol\tau_d - \mathbf C\dot{\boldsymbol\theta} - \mathbf G - \mathbf F\big)
\end{bmatrix}.
$$

The code solves the linear system $\mathbf M\ddot{\boldsymbol\theta} = \mathbf b$ instead of inverting $\mathbf M$.

**Energy balance (used to validate the model).** Using the skew-symmetry property,

$$
\frac{dE}{dt} = \frac{d}{dt}(T + V) = \dot{\boldsymbol\theta}^\top(\boldsymbol\tau + \boldsymbol\tau_d) - \dot{\boldsymbol\theta}^\top\mathbf F(\dot{\boldsymbol\theta}).
$$

With no input torque, energy can only decrease, at exactly the friction dissipation rate. The `validate` case
checks that this balance holds to integrator precision.

### 5.10 Modelling a payload

A point mass $m_p$ at the tool tip is merged with link 3 into one rigid body:

$$
m_3' = m_3 + m_p,\qquad
l_{c3}' = \frac{m_3l_{c3} + m_pl_3}{m_3'},\qquad
I_3' = I_3 + m_3(l_{c3} - l_{c3}')^2 + m_p(l_3 - l_{c3}')^2 \quad(\text{parallel-axis theorem}).
$$

The formulas of §5.5–5.7 are then used unchanged with the primed values. A 1 kg payload doubles the distal mass
and moves its COM to 0.5 m from the wrist.

---

## Part 6 — Trajectory planning

### 6.1 Classical quintic polynomial (joint space)

Each joint follows $q(t) = \sum_{k=0}^{5}a_k\tau^k$ with $\tau = t - t_0 \in [0, T]$. The six coefficients satisfy
position, velocity and acceleration conditions at both ends:

$$
\begin{bmatrix}
1&0&0&0&0&0\\ 0&1&0&0&0&0\\ 0&0&2&0&0&0\\
1&T&T^2&T^3&T^4&T^5\\ 0&1&2T&3T^2&4T^3&5T^4\\ 0&0&2&6T&12T^2&20T^3
\end{bmatrix}\mathbf a =
\begin{bmatrix}q_0\\ \dot q_0\\ \ddot q_0\\ q_f\\ \dot q_f\\ \ddot q_f\end{bmatrix}.
$$

For rest-to-rest motion this reduces to

$$
q(t) = q_0 + (q_f - q_0)\,\big(10s^3 - 15s^4 + 6s^5\big),\qquad s = \tau/T .
$$

> **In plain English:** the quintic is the smoothest simple curve that starts and stops at rest with zero
> acceleration. It ignores the robot's dynamics completely: it does not know about gravity, torque limits or energy.

### 6.2 Optimisation-based trajectory (minimum energy)

**Idea.** Keep the quintic as a "backbone" that satisfies the boundary conditions, and add a correction that
the optimiser can shape freely:

$$
\boldsymbol\theta(s) = \boldsymbol\theta_{\text{quintic}}(s) + \sum_{k=1}^{K}\mathbf c_k\,\psi_k(s),
\qquad
\psi_k(s) = s^3(1 - s)^3\,P_{k-1}(2s - 1),\qquad s = \tau/T \in [0, 1].
$$

$P_{k-1}$ are Legendre polynomials. The factor $s^3(1-s)^3$ makes every basis function vanish **with its first and
second derivatives** at both ends, so any choice of the coefficients $\mathbf c_k \in \mathbb R^3$ keeps the
rest-to-rest boundary conditions. The decision vector is $\mathbf z = [\mathbf c_1, \dots, \mathbf c_K]$ with
$3K$ entries ($K = 5$ by default, i.e. 15 variables).

**Cost function.** The torques along the path come from the nominal inverse dynamics,
$\boldsymbol\tau(t) = \mathbf M\ddot{\boldsymbol\theta} + \mathbf C\dot{\boldsymbol\theta} + \mathbf G + \mathbf F$. We minimise

$$
J_E(\mathbf z) = \int_0^T \boldsymbol\tau(t)^\top\mathbf W\,\boldsymbol\tau(t)\,dt,
\qquad \mathbf W = \operatorname{diag}(1, 1, 1).
$$

For DC motors the copper (resistive) loss is $R\,i^2 \propto \tau^2$, so $J_E$ is proportional to the electrical
energy turned into heat. Unlike the mechanical work $\int|\boldsymbol\tau^\top\dot{\boldsymbol\theta}|\,dt$, it is
also smooth in $\mathbf z$, which suits gradient-based solvers. Both quantities are reported in the results.

**Constraints**, enforced at $N = 61$ collocation points $t_j$:

| Constraint | Meaning |
|---|---|
| $\lvert\tau_i(t_j)\rvert \le \tau_{i,\max}$, with $\tau_{\max} = (60, 30, 10)$ N·m | actuator torque limits |
| $\lvert\dot\theta_i(t_j)\rvert \le \dot\theta_{\max} = 3$ rad/s | actuator speed limits |
| $\theta_{i,\min} \le \theta_i(t_j) \le \theta_{i,\max}$ | joint limits |
| $y$-coordinates of elbow, wrist and tool $\ge y_{\text{floor}} = 0$ | the arm stays above the table |

The boundary conditions are satisfied exactly by construction, so they are not constraints.

**Solver.** Sequential Least-Squares Quadratic Programming (`scipy.optimize.minimize(method="SLSQP")`), starting
from $\mathbf z = \mathbf 0$ (the quintic). The integral is evaluated with Simpson's rule on the collocation grid,
and all dynamics are vectorised over the grid so each cost evaluation is a few array operations.

> **In plain English:** the optimiser bends the quintic path, keeping the same start, end and duration, so that
> the motors work less. In practice it reshapes the motion to lower the gravity torques, for example by
> folding the arm earlier so that the heavy links have shorter lever arms. It does this while respecting torque,
> speed, joint and floor limits.

### 6.3 Minimum-time variant

The duration becomes a decision variable, $\mathbf z' = [\mathbf z,\ T]$, and we minimise

$$
J_T(\mathbf z') = T + \mu\,\frac{J_E(\mathbf z')}{J_E^{\text{ref}}},\qquad \mu = 10^{-3},
$$

subject to the same constraints. The tiny energy term only removes non-uniqueness among equally fast solutions.
Because torques scale like $1/T^2$, the torque limits are what stop $T$ from shrinking further. At the optimum, at
least one actuator touches its limit.

---

## Part 7 — Controllers

Notation: $\mathbf e = \boldsymbol\theta_d - \boldsymbol\theta$ and $\dot{\mathbf e} = \dot{\boldsymbol\theta}_d - \dot{\boldsymbol\theta}$.
Hats ($\hat{\mathbf M}$, $\hat{\mathbf C}$, …) denote the controller's **nominal model**, which differs from
the real plant under uncertainty.

### 7.1 Independent-joint PID

$$
\boxed{\ \boldsymbol\tau = \mathbf K_p\mathbf e + \mathbf K_i\!\int_0^t\!\mathbf e\,d\sigma + \mathbf K_d\dot{\mathbf e}\ }
$$

* Each joint is controlled as if it were alone. Gravity, coupling, Coriolis and friction are treated as
  disturbances that the feedback must reject.
* The integral removes steady-state error at rest. During motion, the slowly varying gravity load still causes
  a tracking lag of order $\Delta G/K_p$.
* Locally, joint $i$ with effective inertia $m_{ii}$ has the characteristic polynomial
  $m_{ii}s^3 + K_{d,i}s^2 + K_{p,i}s + K_{i,i}$. It is stable if $K_{d,i}K_{p,i} > m_{ii}K_{i,i}$ (Routh–Hurwitz).
  Because $m_{11}$ varies about 6×, the damping changes with posture.
* The integrator is clamped (conditional-integration anti-windup).

> **In plain English:** PID knows nothing about the robot. It pushes harder the further behind it is. It is
> simple and robust to model errors, but it is always slightly late, because errors must appear before it reacts.

### 7.2 Computed-torque control with friction compensation

$$
\boxed{\ \boldsymbol\tau = \hat{\mathbf M}(\boldsymbol\theta)\big(\ddot{\boldsymbol\theta}_d + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e\big) + \hat{\mathbf C}(\boldsymbol\theta,\dot{\boldsymbol\theta})\dot{\boldsymbol\theta} + \hat{\mathbf G}(\boldsymbol\theta) + \hat{\mathbf F}(\dot{\boldsymbol\theta})\ }
$$

**With a perfect model**, substituting into the equations of motion and cancelling the invertible $\mathbf M$ gives

$$
\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = -\mathbf M^{-1}\boldsymbol\tau_d .
$$

Without disturbance the error obeys a linear, decoupled, **configuration-independent** second-order system. We
choose $\mathbf K_p = \omega_n^2\mathbf I$ and $\mathbf K_d = 2\zeta\omega_n\mathbf I$ with $\zeta = 1$ (critical damping).

**With model errors** ($\tilde{\mathbf M} = \hat{\mathbf M} - \mathbf M$, etc.):

$$
\ddot{\mathbf e} + \mathbf K_d\dot{\mathbf e} + \mathbf K_p\mathbf e = \boldsymbol\eta,\qquad
\boldsymbol\eta = \hat{\mathbf M}^{-1}\big(\tilde{\mathbf M}\ddot{\boldsymbol\theta} + \tilde{\mathbf C}\dot{\boldsymbol\theta} + \tilde{\mathbf G} + \tilde{\mathbf F} - \boldsymbol\tau_d\big).
$$

The steady error scales like $\|\boldsymbol\eta\|/\omega_n^2$. Note the factor $\hat{\mathbf M}^{-1}$: on the light
distal joints ($\hat M_{33} = 0.06$ kg·m²) a small torque error becomes a large acceleration error.

> **In plain English:** CTC computes the torque the robot *should* need for the desired motion (the inverse
> dynamics, including friction), then adds a small PD correction. If the model is right, the robot behaves like
> three independent, perfectly tuned springs and dampers. If the model is wrong, the leftover error acts as a
> disturbance that the PD correction must fight, and CTC has no built-in mechanism to do that.

### 7.3 Sliding-mode control (SMC)

**Step 1: define a sliding variable.**

$$
\mathbf s = \dot{\mathbf e} + \boldsymbol\Lambda\,\mathbf e,\qquad \boldsymbol\Lambda = \operatorname{diag}(\lambda_i) > 0 .
$$

On the **sliding surface** $\mathbf s = \mathbf 0$ the error obeys $\dot e_i = -\lambda_ie_i$, so it decays
exponentially with time constant $1/\lambda_i$, *whatever the robot's dynamics*. The control problem reduces to
forcing $\mathbf s \to \mathbf 0$.

**Step 2: control law (Slotine–Li form).** Define the reference velocity
$\dot{\boldsymbol\theta}_r = \dot{\boldsymbol\theta}_d + \boldsymbol\Lambda\mathbf e$, so that
$\mathbf s = \dot{\boldsymbol\theta}_r - \dot{\boldsymbol\theta}$, and

$$
\boxed{\ \boldsymbol\tau = \underbrace{\hat{\mathbf M}\ddot{\boldsymbol\theta}_r + \hat{\mathbf C}\dot{\boldsymbol\theta}_r + \hat{\mathbf G} + \hat{\mathbf F}}_{\text{equivalent (model-based) control}}
\;+\; \underbrace{\mathbf K_D\,\mathbf s}_{\text{linear damping}}
\;+\; \underbrace{\mathbf K\,\operatorname{sat}(\mathbf s/\Phi)}_{\text{robust switching term}}\ }
$$

**Step 3: stability proof.** Substituting into the equations of motion and using $\mathbf C\dot{\boldsymbol\theta} = \mathbf C\dot{\boldsymbol\theta}_r - \mathbf C\mathbf s$:

$$
\mathbf M\dot{\mathbf s} + \mathbf C\mathbf s = -\mathbf K_D\mathbf s - \mathbf K\operatorname{sgn}(\mathbf s) + \boldsymbol\Delta,
\qquad
\boldsymbol\Delta = \tilde{\mathbf M}'\ddot{\boldsymbol\theta}_r + \tilde{\mathbf C}'\dot{\boldsymbol\theta}_r + \tilde{\mathbf G}' + \tilde{\mathbf F}' - \boldsymbol\tau_d,
$$

where primes denote the true-minus-model differences. Take $V = \tfrac12\mathbf s^\top\mathbf M\mathbf s$. Using the
skew-symmetry of $\dot{\mathbf M} - 2\mathbf C$:

$$
\dot V = \mathbf s^\top(\mathbf M\dot{\mathbf s} + \mathbf C\mathbf s)
= -\mathbf s^\top\mathbf K_D\mathbf s - \sum_i k_i|s_i| + \mathbf s^\top\boldsymbol\Delta
\;\le\; -\mathbf s^\top\mathbf K_D\mathbf s - \sum_i\big(k_i - |\Delta_i|\big)|s_i| .
$$

If the **switching gain dominates the uncertainty**, $k_i \ge |\Delta_i| + \eta_i$ with $\eta_i > 0$, then
$\dot V < 0$ and $\mathbf s$ reaches zero in finite time. After that, $\mathbf e \to \mathbf 0$ exponentially.
This holds for *any* uncertainty and disturbance within the bound, which is the core robustness guarantee of SMC.

**Step 4: chattering and the boundary layer.** The ideal law uses $\operatorname{sgn}(\mathbf s)$. In a digital
controller this makes the torque jump between $\pm k_i$ every sample, a high-frequency switching called
**chattering** that excites unmodelled dynamics and wears actuators. Replacing it with a saturation inside a
boundary layer of width $\Phi$,

$$
\operatorname{sat}(s_i/\Phi) =
\begin{cases} s_i/\Phi, & |s_i| \le \Phi\\ \operatorname{sgn}(s_i), & |s_i| > \Phi\end{cases}
$$

makes the control continuous. The price is that $\mathbf s$ is only driven *into* the layer, giving the ultimate
bound $|e_i| \le \Phi/\lambda_i$ instead of exact convergence. The code supports `"sign"`, `"sat"` and `"tanh"`
switching functions, and the chattering study compares them.

**Gain selection.** $\boldsymbol\Lambda$ sets the convergence rate on the surface, and $\mathbf K_D$ adds linear
damping. $\mathbf K$ is set from a bound on $|\boldsymbol\Delta|$ for the largest expected payload (1 kg) plus the
disturbance amplitude: mainly the extra gravity torque, about $9.81 \cdot m_p \cdot r$ on each joint, where $r$ is
the lever arm to the tool. $\Phi$ trades chattering against accuracy.

> **In plain English:** SMC combines the model-based feed-forward of CTC with a strong "corrective push" whose
> size is chosen to beat the worst-case model error. As long as the real robot stays within the assumed
> uncertainty, the tracking error is guaranteed to shrink. The push switches direction depending on which side
> of the target the robot is, and the boundary layer softens that switch so the motors are not hammered.

---

## Part 8 — Uncertainty, disturbances and performance metrics

### 8.1 Test scenarios

All controllers use the **nominal** model. The plant is changed as follows:

| Scenario | Plant changes |
|---|---|
| Nominal | none |
| Payload +0.5 kg | 0.5 kg point mass at the tool tip |
| Payload +1.0 kg | 1.0 kg point mass at the tool tip |
| Disturbance | external torque $\boldsymbol\tau_d(t)$ (below) |
| Combined | 1.0 kg payload, link masses +10 %, viscous and Coulomb friction +50 %, **and** $\boldsymbol\tau_d(t)$ |

The disturbance is a sum of sinusoids and short smooth pulses (impacts), applied at the joints:

$$
\tau_{d,i}(t) = A_i\sin(2\pi f_it + \varphi_i) + \sum_k P_{ik}\,\Pi_k(t),
$$

where $\Pi_k$ is a pulse with smooth $\tanh$ edges ($\approx 10$ ms rise time).

### 8.2 Metrics

For a run sampled every $\Delta t$ on $[0, t_f]$ with $N$ samples:

| Metric | Definition |
|---|---|
| Joint RMSE | $\sqrt{\tfrac1N\sum_k e_i(t_k)^2}$ |
| Max error | $\max_k\lvert e_i(t_k)\rvert$ |
| Tool-tip RMSE / max | the same for $\lVert\mathbf p_d - \mathbf p\rVert$ |
| RMS / peak torque | $\sqrt{\tfrac1N\sum_k\tau_i^2}$ and $\max_k\lvert\tau_i\rvert$ |
| Mechanical energy | $E_{\text{mech}} = \int_0^{t_f}\sum_i\lvert\tau_i\dot\theta_i\rvert\,dt$ (no regeneration) |
| Copper-loss proxy | $E_{\tau^2} = \int_0^{t_f}\boldsymbol\tau^\top\boldsymbol\tau\,dt$ |
| Robustness | worst-case and mean tool-tip RMSE over all scenarios, and the RMSE growth from nominal to combined |
| Chattering (total variation) | $\mathrm{TV} = \tfrac{1}{t_f}\sum_k\lVert\boldsymbol\tau_{k+1} - \boldsymbol\tau_k\rVert_1$ [N·m/s] |
| Chattering (spectral) | fraction of the torque signal's power above 20 Hz |

> **In plain English:** RMSE is the typical tracking error and max error the worst moment. Energy says how hard
> the motors worked. Robustness is how much performance degrades when the robot is not what the controller
> thinks it is. Chattering measures how "nervous" the torque signal is: a smooth signal has low total variation
> and almost no high-frequency content.

### 8.3 Simulation method

* **Continuous-time** runs (default) integrate the closed loop with `scipy.integrate.solve_ivp` (RK45,
  `rtol = 1e-7`, `atol = 1e-9`). Controller states such as the PID integral are part of the ODE state, which keeps
  them consistent with the adaptive step-size control.
* **Sampled-data** runs (chattering study) model a digital controller. Torque is computed every
  $T_s = 1$ ms and held constant (zero-order hold), and `solve_ivp` integrates the plant between samples. This is
  the setting in which a $\operatorname{sgn}$-based SMC actually chatters.
