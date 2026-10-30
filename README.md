# FPGA-Based Dual-Branch 2D-CNN Accelerator

## Project Overview

This project focuses on the RTL design and FPGA implementation of hardware modules from a dual-branch 2D Convolutional Neural Network (CNN) architecture for ECG feature processing.

The design explores three hardware optimisation techniques:

- **Parallel-Quantized-Pixel-wise (PQP):** Enables parallel pixel-wise convolution across multiple kernels.
- **Skip-Zero-Weight (SZW):** Skips unnecessary multiply-accumulate operations when the corresponding kernel weight is zero.
- **Fused Add-Pool Unit:** Combines feature maps from two branches through element-wise addition followed by max pooling.

The hardware modules are developed using Verilog HDL and evaluated using Vivado, targeting the PYNQ-Z2 FPGA platform.

## Objectives

- Implement selected dual-branch CNN processing modules using Verilog RTL.
- Develop parallel convolution processing using the PQP architecture.
- Reduce unnecessary MAC operations using SZW logic.
- Combine addition and max pooling into a single functional unit.
- Integrate the custom accelerator IP with the Zynq Processing System through AXI.
- Evaluate FPGA resource utilisation and processing cycle count against standard sequential convolution.

## Architecture

The proposed processing architecture consists of:

1. Input feature-map and weight storage.
2. Window generation and pixel distribution.
3. Parallel convolution using PQP engines.
4. Zero-weight detection and MAC enable control.
5. Feature-map addition.
6. 2 × 2 max pooling.
7. Output buffering.

### Hardware Optimisation Techniques

**PQP Convolution**

Processes shared input pixels in parallel across multiple convolution kernels, reducing repeated input access and enabling higher hardware parallelism.

**Skip-Zero-Weight**

Checks kernel weights before MAC execution. When a weight is zero, the corresponding computation is skipped.

**Fused Add-Pool**

Combines the outputs of both branches through element-wise addition and performs max pooling to reduce spatial dimensions.

## Hardware Platform and Tools

- FPGA Board: PYNQ-Z2
- FPGA Device: [Add exact device part number]
- HDL: Verilog
- Design and Simulation: Xilinx Vivado [Add version]
- Hardware Interface: AXI / Zynq Processing System
- Software: Python [Add version, if applicable]

## FPGA Resource Utilisation

| Resource | PQP + SZW | Standard Convolution |
|---|---:|---:|
| LUTs | 2,616 | 1,885 |
| Registers | 4,596 | 4,501 |
| DSPs | 30 | 16 |
| BRAM tiles | 100.5 | 99.5 |

## Performance Evaluation

| Metric | PQP + SZW | Standard Convolution |
|---|---:|---:|
| Processing cycles | 24.8 million | 170.8 million |
| Estimated time at 100 MHz | 248 ms | 1,707 ms |

The PQP + SZW architecture achieves approximately 6.89× faster simulated processing, corresponding to an approximately 85.5% reduction in processing cycles.

This improvement is obtained at the cost of increased LUT and DSP utilisation.

## Verification

The design was evaluated through:

- RTL functional simulation.
- Pixel and kernel-weight verification.
- Convolution output verification.
- Feature-map generation.
- Add-Pool output verification.
- Vivado FPGA resource utilisation analysis.
- Cycle-count comparison with standard sequential convolution.

## Repository Structure

- `rtl/` – Verilog RTL modules.
- `testbench/` – Simulation testbenches.
- `vivado/` – FPGA integration files and configuration.
- `python/` – Supporting Python scripts, if applicable.
- `data/` – Sample input data and instructions.
- `results/` – Simulation outputs and performance reports.
- `docs/` – Architecture and implementation documentation.
- `images/` – Architecture diagrams and experimental setup.

## Scope and Implementation Status

This repository documents the hardware modules developed and evaluated as part of the project. The software CNN training and classification results are presented separately from the RTL hardware evaluation.

The reported hardware cycle counts are simulation-based estimates at 100 MHz; they should not be interpreted as measured end-to-end application latency.

## References

1. PhysioNet – MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
2. HLS-compiled PYNQ-based cardiac arrhythmia detection system leveraging quantized ECG beat images https://www.sciencedirect.com/science/article/pii/S1746809425005749
3. AMD/Xilinx Vivado Design Suite documentation: https://docs.amd.com/

## Acknowledgement

Developed as part of the EC23702 Mini Project, Department of Electronics and Communication Engineering, College of Engineering Guindy, Anna University.
We are immensely grateful to our mini-project supervisor, Dr.C.M Sujatha, for her valuable guidance, technical mentorship, patience, and continuous encouragement 
throughout the development of this project.
