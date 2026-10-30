# Project Overview

## FPGA-Based Dual-Branch 2D-CNN Accelerator

### 1. Introduction

Convolutional Neural Networks (CNNs) are widely used for image-based feature extraction and classification. However, convolution operations involve a large number of multiply-accumulate (MAC) operations, which can increase computational latency when executed sequentially.

This project focuses on the RTL design and FPGA-based implementation of an optimised dual-branch 2D-CNN processing architecture. The design explores parallel convolution, zero-weight computation skipping and feature-map merging to improve hardware execution performance.

The architecture is developed using Verilog HDL and evaluated using Xilinx Vivado, targeting the PYNQ-Z2 FPGA development board.

### 2. Problem Statement

Conventional sequential convolution reuses a limited number of arithmetic resources to perform successive pixel-weight multiplication and accumulation operations. Although this reduces hardware requirements, it can result in a high number of processing cycles.

The project investigates a hardware architecture that improves convolution execution through parallel processing and the elimination of unnecessary zero-weight computations.

### 3. Project Objectives

- Design a dual-branch CNN processing architecture using Verilog HDL.
- Implement Parallel-Quantized-Pixel-wise (PQP) convolution.
- Incorporate Skip-Zero-Weight (SZW) logic to avoid unnecessary computations.
- Develop a fused Add-Pool unit for branch merging and spatial downsampling.
- Verify the RTL modules through simulation.
- Integrate the custom accelerator IP with the Zynq Processing System.
- Evaluate FPGA resource utilisation and processing cycle count.
- Compare the proposed PQP + SZW implementation against standard sequential convolution.

### 4. Proposed Approach

The proposed architecture consists of two convolution branches that extract features using different kernel configurations.

- **Branch 1:** Uses 3 × 3 convolution processing.
- **Branch 2:** Uses a 5 × 5 Sobel kernel for directional feature extraction.

The outputs are combined using an Add-Pool unit, which performs element-wise addition followed by 2 × 2 max pooling.

The PQP architecture enables parallel pixel-wise computation, while SZW skips multiplication operations associated with zero-valued weights.

### 5. Hardware Platform and Tools

| Category | Technology |
|---|---|
| Hardware description language | Verilog HDL |
| FPGA development platform | PYNQ-Z2 |
| FPGA design and simulation | Xilinx Vivado |
| Processing System | Zynq-7000 |
| Hardware interface | AXI |
| Supporting software | Python, where applicable |

### 6. Project Scope

The hardware implementation focuses on convolution processing, feature-map generation, branch merging, pooling and FPGA resource/performance evaluation.

The software CNN model development and its classification results are treated separately from the RTL hardware evaluation. The reported hardware cycle counts represent simulation-based processing estimates rather than measured end-to-end classification latency.

### 7. Expected Outcomes

- Functional RTL implementation of the proposed processing modules.
- Verification of convolution and feature-map processing.
- Integration of custom hardware IP with the Zynq Processing System.
- Evaluation of FPGA resource utilisation.
- Performance comparison against standard sequential convolution.

### 8. Repository Contents

The repository is organised into the following sections:

- `rtl/` – Verilog hardware modules.
- `testbench/` – RTL verification testbenches.
- `python/` – Supporting software scripts, if included.
- `vivado/` – FPGA integration and configuration files.
- `results/` – Simulation outputs and evaluation results.
- `docs/` – Project architecture and technical documentation.
- `images/` – Architecture diagrams and experimental setup images.
