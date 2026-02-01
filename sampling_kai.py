import cudaq
from math import sin, pi

cudaq.set_target("nvidia")
# cudaq.set_target("nvidia-mgpu")

def compute_topology_overlaps(G2, G4):
    """
    Computes the topological invariants I_22, I_24, I_44 based on set overlaps.
    I_alpha_beta counts how many sets share IDENTICAL elements.
    """

    # Helper to count identical sets
    def count_matches(list_a, list_b):
        matches = 0
        # Convert to sorted tuples to ensure order doesn't affect equality
        set_b = set(tuple(sorted(x)) for x in list_b)
        for item in list_a:
            if tuple(sorted(item)) in set_b:
                matches += 1
        return matches

    # For standard LABS/Ising chains, these overlaps are often 0 or specific integers
    # We implement the general counting logic here.
    I_22 = count_matches(G2, G2)  # Self overlap is just len(G2)
    I_44 = count_matches(G4, G4)  # Self overlap is just len(G4)
    I_24 = 0  # 2-body set vs 4-body set overlap usually 0 as sizes differ

    return {"22": I_22, "44": I_44, "24": I_24}


def compute_theta(t, dt, total_time, N, G2, G4):
    """
    Computes theta(t) using the analytical solutions for Gamma1 and Gamma2.
    """

    # ---  Better Schedule (Trigonometric) ---
    # lambda(t) = sin^2(pi * t / 2T)
    # lambda_dot(t) = (pi / 2T) * sin(pi * t / T)

    if total_time == 0:
        return 0.0

    # Argument for the trig functions
    arg = (pi * t) / (2.0 * total_time)

    lam = sin(arg) ** 2
    # Derivative: (pi/2T) * sin(2 * arg) -> sin(pi * t / T)
    lam_dot = (pi / (2.0 * total_time)) * sin((pi * t) / total_time)

    # ---  Calculate Gamma Terms (LABS assumptions: h^x=1, h^b=0) ---
    # For G2 (size 2): S_x = 2
    # For G4 (size 4): S_x = 4

    # Gamma 1 (Eq 16)
    # Gamma1 = 16 * Sum_G2(S_x) + 64 * Sum_G4(S_x)
    term_g1_2 = 16 * len(G2) * 2
    term_g1_4 = 64 * len(G4) * 4
    Gamma1 = term_g1_2 + term_g1_4

    # Gamma 2 (Eq 17)
    # G2 term: Sum (lambda^2 * S_x)
    # S_x = 2
    sum_G2 = len(G2) * (lam**2 * 2)

    # G4 term: 4 * Sum (4*lambda^2 * S_x + (1-lambda)^2 * 8)
    # S_x = 4
    # Inner = 16*lam^2 + 8*(1-lam)^2
    sum_G4 = 4 * len(G4) * (16 * (lam**2) + 8 * ((1 - lam) ** 2))

    # Topology part
    I_vals = compute_topology_overlaps(G2, G4)
    term_topology = (
        4 * (lam**2) * (4 * I_vals["24"] + I_vals["22"]) + 64 * (lam**2) * I_vals["44"]
    )

    # Combine Gamma 2
    Gamma2 = -256 * (term_topology + sum_G2 + sum_G4)

    # ---  Alpha & Theta ---
    if abs(Gamma2) < 1e-12:
        alpha = 0.0
    else:
        alpha = -Gamma1 / Gamma2

    return dt * alpha * lam_dot


@cudaq.kernel
def rzz(theta: float, q0: cudaq.qubit, q1: cudaq.qubit):
    cx(q0, q1)
    rz(theta, q1)
    cx(q0, q1)


@cudaq.kernel
def two_qubit_operator(q0: cudaq.qubit, q1: cudaq.qubit, theta: float):
    # Left Rx layer
    rx(math.pi / 2, q1)

    # Entangling gate
    rzz(theta, q0, q1)

    # Right Rx layer
    rx(math.pi / 2, q0)
    rx(-math.pi / 2, q1)

    rzz(theta, q0, q1)

    rx(-math.pi / 2, q0)


@cudaq.kernel
def four_qubit_operator(
    q0: cudaq.qubit, q1: cudaq.qubit, q2: cudaq.qubit, q3: cudaq.qubit, theta: float
):
    # 1st layer
    rx(-math.pi / 2, q0)
    ry(math.pi / 2, q1)
    ry(-math.pi / 2, q2)

    # 2nd layer
    rzz(-math.pi / 2, q0, q1)
    rzz(-math.pi / 2, q2, q3)

    # 3rd layer
    rx(math.pi / 2, q0)
    ry(-math.pi / 2, q1)
    ry(math.pi / 2, q2)
    rx(-math.pi / 2, q3)

    # 4th layer
    rx(-math.pi / 2, q1)
    rx(-math.pi / 2, q2)

    # 5th layer
    rzz(theta, q1, q2)

    # 6th layer
    rx(math.pi / 2, q1)
    rx(math.pi, q2)

    # 7th layer
    ry(math.pi / 2, q1)

    # 8th layer
    rzz(math.pi / 2, q0, q1)

    # 9th layer
    rx(math.pi / 2, q0)
    ry(-math.pi / 2, q1)

    # 10th layer
    rzz(-theta, q1, q2)

    # 11th layer
    rx(math.pi / 2, q1)
    rx(-math.pi, q2)

    # 12th layer
    rzz(-theta, q1, q2)

    # 13th layer
    rx(-math.pi, q1)
    ry(math.pi / 2, q2)

    # 14th layer
    rzz(-math.pi / 2, q2, q3)

    # 15th layer
    ry(-math.pi / 2, q2)
    rx(-math.pi / 2, q3)

    # 16th layer
    rx(-math.pi / 2, q2)

    # 17th layer
    rzz(theta, q1, q2)

    # 18th layer
    rx(math.pi / 2, q1)
    rx(math.pi / 2, q2)

    # 19th layer
    ry(-math.pi / 2, q1)
    ry(math.pi / 2, q2)

    # 20th layer
    rzz(math.pi / 2, q0, q1)
    rzz(math.pi / 2, q2, q3)

    # 21th layer
    ry(math.pi / 2, q1)
    ry(-math.pi / 2, q2)
    rx(math.pi / 2, q3)


def get_interactions(N):
    """
    Generates the interaction sets G2 and G4 based on the loop limits in Eq. 15.
    Returns standard 0-based indices as lists of lists of ints.

    Args:
        N (int): Sequence length.

    Returns:
        G2: List of lists containing two body term indices
        G4: List of lists containing four body term indices
    """

    G2 = []
    G4 = []

    # Two-body terms
    for i in range(N - 2):
        for k in range(1, (N - i) // 2 + 1):
            G2.append([i, i + k])

    # Four-body terms
    for i in range(N - 3):
        for t in range(1, (N - i - 1) // 2 + 1):
            for k in range(t + 1, N - i - t):
                G4.append([i, i + t, i + k, i + k + t])

    return G2, G4


@cudaq.kernel
def trotterized_circuit(
    N: int,
    G2: list[list[int]],
    G4: list[list[int]],
    steps: int,
    dt: float,
    T: float,
    thetas: list[float],
):

    reg = cudaq.qvector(N)
    h(reg)

    # TODO - write the full kernel to apply the trotterized circuit

    for n in range(steps):
        theta_n = thetas[n]
        for i, j in G2:
            angle = 4.0 * theta_n * dt
            two_qubit_operator(reg[i], reg[j], angle)

        for i, j, k, l in G4:
            angle = 8.0 * theta_n * dt
            four_qubit_operator(reg[i], reg[j], reg[k], reg[l], angle)


T = 1  # total time
n_steps = 1  # number of trotter steps
dt = T / n_steps
N = 12

G2, G4 = get_interactions(N)
thetas = []
for step in range(1, n_steps + 1):
    t = step * dt
    theta_val = compute_theta(t, dt, T, N, G2, G4)
    thetas.append(theta_val)


# TODO - Sample your kernel to make sure it works

# Create a sample object
shots = 100  # number of times to run the circuit

result = cudaq.sample(
    trotterized_circuit, N, G2, G4, n_steps, dt, T, thetas, shots_count=shots
)

# Print the results
print(result)