import argparse
import numpy as np
import cudaq
import os

import matplotlib.pyplot as plt
import seaborn as sns
import time
import cupy as cp

import cudaq_solvers as solvers
import torch
from cudaq import spin
from lightning.fabric.loggers import CSVLogger
from src.GQEMTS.gqe import get_default_config

from cudaq_mlir_parser import parse, simulate

from collections import Counter

# ==========================================
# 1. Argument Parsing and Environment Setup
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--mpi", action="store_true", help="Enable MPI distribution")
args = parser.parse_args()

# Initialize CUDA-Q target based on MPI flag
if args.mpi:
    try:
        cudaq.set_target("nvidia", option="mqpu")
        cudaq.mpi.initialize()
    except RuntimeError:
        print(
            "Warning: NVIDIA GPUs or MPI not available, unable to use CUDA-Q MQPU. Skipping..."
        )
        exit(0)
else:
    try:
        cudaq.set_target("nvidia", option="fp64")
    except RuntimeError:
        # Fallback to CPU if NVIDIA target is not available
        cudaq.set_target("qpp-cpu")

# ==========================================
# 2. Imports and Reproducibility Configuration
# ==========================================



# Set deterministic behavior for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.manual_seed(3047)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==========================================
# 3. Hamiltonian Definition (LABS Problem)
# ==========================================

def labs_spin_op(N: int):
    """
    Constructs the Hamiltonian for the Low Autocorrelation Binary Sequence (LABS) problem.
    """
    H = 0.0

    # ---- 2-body interaction terms ----
    for i in range(N - 2):
        max_k = (N - i) // 2
        for k in range(1, max_k + 1):
            H += 2.0 * spin.z(i) * spin.z(i + k)

    # ---- 4-body interaction terms ----
    for i in range(N - 3):
        max_t = (N - i - 1) // 2
        for t in range(1, max_t + 1):
            for k in range(t + 1, N - i - t):
                H += (
                    4.0
                    * spin.z(i)
                    * spin.z(i + t)
                    * spin.z(i + k)
                    * spin.z(i + k + t)
                )

    return H

# Initialize problem parameters
N = 25
spin_ham = labs_spin_op(N)
n_qubits = N

# ==========================================
# 4. Operator Pool Construction
# ==========================================

# Coefficients for the rotation gates
params = [
    0.05,
    -0.05,
    0.1,
    -0.1,
]

def pool(params, n_qubits):
    """
    Generates a pool of variational operators (Ansatz pool).
    Includes 2-qubit and 4-qubit rotation terms.
    """
    ops = []

    # 2-qubit rotations (Nearest Neighbor)
    for i in range(n_qubits - 1):
        ops.append(cudaq.SpinOperator(spin.y(i) * spin.z(i + 1)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.y(i + 1)))

    # 4-qubit rotations
    for i in range(n_qubits - 3):
        ops.append(cudaq.SpinOperator(spin.y(i) * spin.z(i + 1) * spin.z(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.y(i + 1) * spin.z(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1) * spin.y(i + 2) * spin.z(i + 3)))
        ops.append(cudaq.SpinOperator(spin.z(i) * spin.z(i + 1) * spin.z(i + 2) * spin.y(i + 3)))
    
    pool_ops = []

    # Combine base operators with parameters
    for c in params:
        for op in ops:
            pool_ops.append(c * op)

    return pool_ops

# Create the operator pool
op_pool = pool(params, n_qubits)
print("Number of operators in pool:", len(op_pool))

# ==========================================
# 5. Helper Functions & Kernels
# ==========================================

def term_coefficients(op: cudaq.SpinOperator) -> list[complex]:
    """Extract coefficients from a SpinOperator."""
    return [term.evaluate_coefficient() for term in op]


def term_words(op: cudaq.SpinOperator) -> list[cudaq.pauli_word]:
    """Extract Pauli words from a SpinOperator."""
    return [term.get_pauli_word(n_qubits) for term in op]


@cudaq.kernel
def kernel(coeffs: list[float], words: list[cudaq.pauli_word]):
    """
    Quantum Kernel for energy estimation.
    Applies Hadamard gates followed by the parameterized Pauli exponentials.
    """
    q = cudaq.qvector(25)

    # Start from superposition state |+>^N
    for i in range(25):
        h(q[i])

    # Apply parameterized ansatz
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])


def cost(sampled_ops: list[cudaq.SpinOperator], **kwargs):
    """
    Cost function to evaluate the expectation value of the Hamiltonian.
    Handles both MPI (async) and standard execution.
    """
    full_coeffs = []
    full_words = []

    # Flatten the operators into coefficients and words for the kernel
    for op in sampled_ops:
        full_coeffs += [c.real for c in term_coefficients(op)]
        full_words += term_words(op)

    if args.mpi:
        handle = cudaq.observe_async(
            kernel,
            spin_ham,
            full_coeffs,
            full_words,
            qpu_id=kwargs["qpu_id"],
        )
        # Return handle and a lambda to retrieve the result later
        return handle, lambda res: res.get().expectation()
    else:
        return cudaq.observe(
            kernel, spin_ham, full_coeffs, full_words
        ).expectation()


@cudaq.kernel
def sample_optimized(coeffs: list[float], words: list[cudaq.pauli_word]):
    """
    Quantum Kernel for sampling the final state.
    Similar to 'kernel' but includes measurement (mz).
    """
    q = cudaq.qvector(n_qubits)

    # Start from |+>^N
    for i in range(n_qubits):
        h(q[i])

    # Apply the optimized evolution
    for i in range(len(coeffs)):
        exp_pauli(coeffs[i], q, words[i])

    # Measure all qubits in the Z-basis
    for qubit in q:
        mz(qubit)


def labs_energy(x):
    """
    Calculates the classical LABS energy for a given bitstring configuration x.
    Used to verify the quantum result.
    """
    s = 2 * np.asarray(x) - 1  # Convert 0/1 to -1/+1
    N = len(s)
    E = 0
    for k in range(1, N):
        Ck = np.sum(s[:N - k] * s[k:])
        E += Ck**2
    return E

# ==========================================
# 6. Main Execution (GQE Solver)
# ==========================================

# Configure GQE (Genetic Quantum Eigensolver) settings
cfg = get_default_config()
cfg.use_fabric_logging = False
logger = CSVLogger("gqe_h2_logs/gqe.csv")
cfg.fabric_logger = logger
cfg.save_trajectory = False
cfg.verbose = True


initial_time=time.time()
# Run the GQE solver
minE, best_ops = solvers.gqe(cost, op_pool, max_iters=5, ngates=3, config=cfg)

# ==========================================
# 7. Results and Post-Processing
# ==========================================

def get_mts_population(counts, n_agents, n_bits):
    """
    將 cudaq 的採樣結果轉換為 MTS 需要的 (N_AGENTS, N_BITS) numpy int8 矩陣。
    """
    
    # 1. 取出前 n_agents 個最頻繁的結果
    # top_results 是一個 list of tuples: [('0010...', 4), ('110...', 3), ...]
    top_results = counts.most_common(n_agents)
    found_count = len(top_results)
    
    # 2. 初始化全零矩陣 (Host Memory / CPU)
    # dtype=np.int8 是為了配合先前 C++ Kernel 的 signed char 要求
    population = np.zeros((n_agents, n_bits), dtype=np.int8)
    
    # 3. 填入量子採樣的結果
    for i in range(found_count):
        bitstring = top_results[i][0]  # 取得位元串，例如 '001001111100'
        
        # 防呆檢查：確保長度正確
        if len(bitstring) != n_bits:
            # 有些模擬器可能會省略前導 0，這邊做補齊 (視情況而定，通常 cudaq 會給定長)
            bitstring = bitstring.zfill(n_bits)
            
        # 將字串轉為整數陣列
        population[i] = np.array([int(c) for c in bitstring], dtype=np.int8)
        
    # 4. 如果找到的唯一解數量 < n_agents，剩下的用隨機填補
    # 這是為了保持矩陣形狀完整，避免 MTS 出錯
    if found_count < n_agents:
        print(f"Warning: Only found {found_count} unique solutions. Filling {n_agents - found_count} agents with random bits.")
        remaining = n_agents - found_count
        population[found_count:] = np.random.randint(2, size=(remaining, n_bits), dtype=np.int8)
        
    return population

# Only print results from the root rank if using MPI
if not args.mpi or cudaq.mpi.rank() == 0:
    print(f"Ground Energy = {minE}")
    print("Ansatz Ops")
    
    opt_coeffs = []
    opt_words = []
    
    # Process the best operators found by GQE
    for idx in best_ops:
        # Print the first term of the operator for inspection
        term = next(iter(op_pool[idx]))
        print(term.evaluate_coefficient().real, term.get_pauli_word(n_qubits))
        
        # Collect all terms for the final sampling circuit
        op = op_pool[idx]
        for term in op:
            opt_coeffs.append(term.evaluate_coefficient().real)
            opt_words.append(term.get_pauli_word(n_qubits))

    shots = 10000

    # Execute the optimized circuit to get samples
    samples = cudaq.sample(sample_optimized, opt_coeffs, opt_words, shots_count=shots)

    # Process sample results

    circuit = parse(kernel)
    print("  qubits =", circuit.num_qubits)
    print("  gates  =", len(circuit.gates))
    state = simulate(circuit)
    probs = np.abs(state) ** 2

    energies = []
    bitsrings= []
    for i, p in enumerate(probs):
        if p < 1e-6:
            continue
        bit = [int(c) for c in f"{i:0{n_qubits}b}"]
        energies.append(labs_energy(bit))
        bitsrings.append(bit)
    gpu_population = cp.asarray(bitsrings, dtype=cp.int8)

    MTS_N_AGENTS=100

    # counts = Counter(dict(samples.items()))

    # MTS_N_AGENTS=100

    # print(f"{MTS_N_AGENTS} most frequent bitstrings:")
    # print(counts.most_common(MTS_N_AGENTS))

    # # Calculate exact classical energies for the sampled bitstrings
    # sample_energies = [labs_energy([int(c) for c in b]) for b in counts.keys()]
    # print("Minimum sampled LABS energy:", min(sample_energies))

    # initial_population = get_mts_population(counts, MTS_N_AGENTS, n_qubits)

    # gpu_population = cp.asarray(initial_population, dtype=cp.int8)
    # print("Initial population for MTS (on GPU):")
    # print(gpu_population)

# --- CUDA Kernel Definition ---
# This C++ Kernel is compiled and executed directly on the GPU.
# Each GPU thread handles one Agent, performing the entire autocorrelation (Lag) 
# calculation within the Registers/L1 Cache for maximum throughput.
labs_energy_kernel_code = r'''
extern "C" __global__
void labs_energy(const signed char* __restrict__ population,
                 float* __restrict__ energies,
                 int n_agents, int n_bits) {

    // 1. Calculate unique Thread ID (corresponds to Agent Index)
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_agents) return;

    // 2. Load the Agent's sequence into Local Memory (Registers/L1 Cache)
    // Assuming N_BITS <= 128 (LABS problems are typically < 100).
    // This avoids repeated Global Memory access, significantly boosting speed.
    const int MAX_BITS = 128;
    signed char seq[MAX_BITS];

    int offset = tid * n_bits;

    // Load and Transform: Convert binary (0, 1) to spin (-1, +1)
    // We do this on the fly to save VRAM and bandwidth.
    for (int i = 0; i < n_bits; i++) {
        seq[i] = 2 * population[offset + i] - 1;
    }

    float total_energy = 0.0f;

    // 3. Calculate Autocorrelation (Energy) for all Lags
    // The logic originally in Python loops is now "Fused" into this single Kernel.
    for (int k = 1; k < n_bits; k++) {
        int correlation = 0;
        for (int i = 0; i < n_bits - k; i++) {
            correlation += seq[i] * seq[i+k];
        }
        // Energy E(S) = sum of squares of autocorrelations
        total_energy += (float)(correlation * correlation);
    }

    // 4. Write the result back to global memory
    energies[tid] = total_energy;
}
'''

class LabsMTS_GPU_Opt:
    def __init__(self, n_bits, n_agents, max_iter):
        self.n_bits = n_bits
        self.n_agents = n_agents
        self.max_iter = max_iter

        # Compile the CUDA Kernel
        self.energy_kernel = cp.RawKernel(labs_energy_kernel_code, 'labs_energy')

    def calculate_labs_energy_batch(self, population):
        """
        Executes the custom CUDA Kernel for high-speed energy calculation.
        """
        n_agents = population.shape[0]
        energies = cp.zeros(n_agents, dtype=cp.float32)

        # Configure Grid and Block dimensions
        threads_per_block = 256
        blocks_per_grid = (n_agents + threads_per_block - 1) // threads_per_block

        # Launch Kernel
        # Arguments: (population_ptr, output_energies_ptr, n_agents, n_bits)
        self.energy_kernel((blocks_per_grid,), (threads_per_block,),
                           (population, energies, n_agents, self.n_bits))

        return energies

    def get_neighbor_batch(self, population, perturbation_strength=1):
        """
        Optimized Neighbor Generation.
        Creates a copy of the population and flips bits based on perturbation strength.
        """
        n_agents, n_bits = population.shape
        neighbor = population.copy()

        # Create row indices for advanced indexing
        rows = cp.arange(n_agents).reshape(-1, 1)

        if perturbation_strength == 1:
            # Local Search: Randomly flip exactly 1 bit per agent.
            # This operation is extremely fast.
            flip_indices = cp.random.randint(0, n_bits, size=(n_agents, 1))
            neighbor[rows, flip_indices] = 1 - neighbor[rows, flip_indices]

        else:
            # Wide Search: Randomly flip 'k' bits per agent.
            # Optimization: Avoid fully sorting a random noise matrix.
            # Instead, directly generate 'k' random indices.
            # Note: There is a small chance of collision (flipping the same bit twice = no change),
            # but this stochastic behavior is acceptable for random search and much faster.
            
            flip_indices = cp.random.randint(0, n_bits, size=(n_agents, perturbation_strength))

            # Apply flips. 
            # We loop 'k' times because 'k' is small (usually < 5). 
            # This is more memory-efficient than handling large index arrays.
            for k in range(perturbation_strength):
                idx = flip_indices[:, k]
                # Toggle bit: 0->1, 1->0
                neighbor[cp.arange(n_agents), idx] = 1 - neighbor[cp.arange(n_agents), idx]

        return neighbor

    def run(self):
        print(f"Starting Optimized MTS (Custom CUDA) for LABS (N={self.n_bits}) with {self.n_agents} agents...")

        # --- INITIALIZATION ---
        # Memory Optimization: Use int8 (matches C++ signed char).
        # This is where the initial random strings are generated.
        population = gpu_population
        print("Initial population for MTS (on GPU):")
        print(population)
        
        
        # Calculate initial energies
        energies = self.calculate_labs_energy_batch(population)

        # Find initial best
        best_idx = cp.argmin(energies)
        global_best_solution = population[best_idx].copy()
        global_best_energy = energies[best_idx]

        history_best_energy = []
        current_global_best_cpu = float(global_best_energy)
        print(f"Initial Best Energy: {current_global_best_cpu}")

        start_time = time.time()

        # Strategy Parameter: Wide search strength
        strength_2 = max(2, int(self.n_bits * 0.05))

        for it in range(self.max_iter):
            # --- MTS Search Strategy ---

            # 1. Generate Candidates (Local Search - flip 1 bit)
            candidate_1 = self.get_neighbor_batch(population, perturbation_strength=1)
            e1 = self.calculate_labs_energy_batch(candidate_1)

            # 2. Update agents that improved
            improved_mask = e1 < energies
            population[improved_mask] = candidate_1[improved_mask]
            energies[improved_mask] = e1[improved_mask]

            # 3. Wide Search for agents that did NOT improve
            # We apply a larger perturbation to escape local optima.
            not_improved_mask = ~improved_mask
            
            # Only proceed if there are agents needing wide search
            if cp.any(not_improved_mask):
                candidate_2 = self.get_neighbor_batch(population, perturbation_strength=strength_2)
                e2 = self.calculate_labs_energy_batch(candidate_2)

                # Update only if (Improved AND Was_Stuck)
                improved_s2_mask = (e2 < energies) & not_improved_mask
                population[improved_s2_mask] = candidate_2[improved_s2_mask]
                energies[improved_s2_mask] = e2[improved_s2_mask]

            # 4. Update Global Best Solution
            current_best_idx = cp.argmin(energies)
            current_min_energy = energies[current_best_idx]
            if current_min_energy ==36:
                final_time = time.time() - initial_time
                print("final time:", final_time)


            # Trigger CPU-GPU synchronization only when updating best to keep logs
            if current_min_energy < global_best_energy:
                global_best_energy = current_min_energy
                global_best_solution = population[current_best_idx].copy()
                current_global_best_cpu = float(global_best_energy)
                print(f"Iter {it}: New Best: {current_global_best_cpu}")

            history_best_energy.append(current_global_best_cpu)

        # Final synchronization for timing
        cp.cuda.Stream.null.synchronize()
        total_time = time.time() - start_time
        print(f"Optimization finished in {total_time:.2f} seconds. ({(self.max_iter/total_time):.1f} it/s)")

        return (cp.asnumpy(global_best_solution),
                float(global_best_energy),
                cp.asnumpy(energies),
                history_best_energy)

# Finalize MPI if necessary
if args.mpi:
    cudaq.mpi.finalize()

if __name__ == "__main__":
    # N_BITS = 38
    # N_AGENTS = 5000000  # 5 million Agents
    MAX_ITER = 100


    mts_opt = LabsMTS_GPU_Opt(n_qubits, MTS_N_AGENTS, MAX_ITER)
    best_sol, best_energy, final_energies, history = mts_opt.run()

    

    print(f"Final Best Energy: {best_energy}")

    print(f"Total time including GQE and MTS: {final_time:.2f} seconds.")