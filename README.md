# NVIDIA iQuHACK 2026 Challenge

## Overview

The Low Autocorrelation of Binary Sequences (LABS) problem is a notoriously difficult optimization challenge, critical for high-performance radar and telecommunications.

Your objective is to take the current classical state-of-the-art Memetic Tabu Search (MTS) and evolve it. Rather than jumping to a purely quantum solution, you will engineer a hybrid quantum-enhanced workflow where samples from a quantum algorithm are used to seed the classical MTS population. You must then push the limits of performance by GPU-accelerating both the quantum simulation and the classical search components.

**We want you to vibe code!**
In modern R&D and this challenge, speed matters, but rigor and coordination matter more. We expect you to employ Agentic Strategies, utilizing AI tools that can reason across your codebase to act as your collaborators while you operate as the Technical Leadership Team. Your collective job is to decompose the problem, delegate tasks across your team and AI agents, and most importantly verify the work. As Leads, you must clearly communicate your planning, workflow, and solution, ensuring your team remains aligned and ready to pivot even as technical challenges shift your strategy.

**Milestones and Evaluation**

This challenge is divided into four milestones, all of which are graded components of your final submission:

1. Ramp Up: Master the state-of-the-art for LABS via a scaffolded tutorial.
2. Research & Plan: Perform due diligence to design a custom quantum strategy and acceleration plan.
3. Build: Validate your algorithm on a CPU in qbraid, then migrate to Brev.dev to deploy full GPU acceleration.
4. Showcase and Retrospective: Present your solution, performance metrics, and your AI-driven workflow.

## Logistics

In this challenge, you will mimic a real-world R&D pipeline, moving from rapid prototyping to high-performance deployment. You will utilize two distinct platforms, each chosen for a specific phase of your development lifecycle:

* **Phase 1 (Prototyping): qBraid**

    For the "Ramp Up" and initial CPU validation, you will work on Milestones 1 and 2 in qBraid. This is your "Dev Environment"—a zero-setup, pre-configured cudaq sandbox that allows you to focus entirely on mastering the algorithm and logic without worrying about infrastructure overhead.

* **Phase 2 (Acceleration): Brev**

    Once your logic is validated, you will "graduate" your code to [Brev](https://brev.nvidia.com/) to complete Milestone 3 and 4. Brev provides on-demand access to a wide variety of NVIDIA GPU architectures (L4s, T4s, A100s, ...). You will use this platform to test your solution across different hardware configurations and unlock full GPU acceleration.

## Phase 1 (Prototyping) on the qBraid Platform

During the duration of the hackathon, you will have access to the new and improved version of the qBraid platform accessible through here: https://account-v2.qbraid.com/

## Phase 2 (Acceleration) on the Brev Platform

Congratulations on finishing the first part of the challenge! 

You will now get to run your code on real GPUs using NVIDIA's Brev Platform. Don't worry, you don't need to pay for anything! Once completing phase 1 and your logic is validated, we will provide you with a **$20 Brev coupon code**. 

> Take budget into consideration when selecting a GPU to run your code on. We know it's tempting to select a B300, but selecting more expensive options will burn through your credits significantly faster.

## Resources
### CUDA-Q    

* If you are new to quantum computing, (e.g., you'd like a review of the definition of a qubit, quantum gates, and quantum circuits), then run through the first two notebooks of the [Quick Start to Quantum Computing](https://github.com/NVIDIA/cuda-q-academic/tree/main/quick-start-to-quantum) series.

* If you have quantum computing background and want a quick visual guide for translating a quantum circuit into a CUDA-Q kernel, check out this [hello world visualization tool](https://nvidia.github.io/cuda-q-academic/quick-start-to-quantum/interactive_widget/cudaq-hello-world.html).  For more in depth coverage of cuda-q syntax and examples, we recommend notebook 1 of the [QAOA for Max Cut series](https://github.com/NVIDIA/cuda-q-academic/tree/main/qaoa-for-max-cut) series as well as the [examples](https://nvidia.github.io/cuda-quantum/latest/using/examples/examples.html) and [applications](https://nvidia.github.io/cuda-quantum/latest/using/applications.html) in the CUDA-Q documentation, in particular the [QAOA example](https://nvidia.github.io/cuda-quantum/latest/applications/python/qaoa.html).  

* If you prefer to learn by watching, you can check out [minutes 30:40-38:28 of this demo](https://www.nvidia.com/en-us/on-demand/session/gtcdc25-dct51159/?playlistId=gtcdc25-quantum-computing-and-hpc&start=1840&end=2308) of description of cudaq kernels, sampling, and getting the state vector or watch [minutes 9:20-19:00 of this demo](https://www.youtube.com/live/DqPC-nlcXKA?si=ualhUnFYjW9BlbQz&t=560).

### qbraid 

### Brev 

### Quantum computing in general

## Accessing Material Post Challenge

Challenge materials can be accessed via https://account-v2.qbraid.com/. 

If you completed Phase 2 of the challenge, materials can be accessed via https://brev.nvidia.com/.
