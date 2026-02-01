import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import cupy as cp


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
        population = cp.random.randint(2, size=(self.n_agents, self.n_bits), dtype=cp.int8)
        
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

if __name__ == "__main__":
    N_BITS = 38
    N_AGENTS = 5000000  # 5 million Agents
    MAX_ITER = 100


    mts_opt = LabsMTS_GPU_Opt(N_BITS, N_AGENTS, MAX_ITER)
    best_sol, best_energy, final_energies, history = mts_opt.run()

    print(f"Final Best Energy: {best_energy}")