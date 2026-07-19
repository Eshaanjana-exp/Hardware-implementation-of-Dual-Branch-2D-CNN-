# Hardware-implementation-of-Dual-Branch-2D-CNN-
Deploying computationally intense Deep Learning models like Convolutional Neural Networks on a Field-Programmable Gate Array 
(FPGA) remains a challenge. To address this, a hardware-aware Dual-Branch 2D-CNN architecture is proposed to enhance the FPGA’s 
inherent parallelism and minimize data-reuse complexity. The architecture will process 4-bit quantized inputs through three key custom 
micro-architectures: a parallel-quantized-pixel-wise (PQP) convolution module to maximize parallelism, a function-merging add-pool unit 
to reduce memory dependency, and a skip-zero-weight (SZW) architecture to eliminate multiply-accumulate (MAC) computations for 
zeroed filter weights, thereby minimizing DSP usage. This design completely eliminates ReLU activations by leveraging unsigned 4-bit 
integer (UINT4) calculations to ensure non-negative intermediate results. Then the optimized design can be exported as a custom RTL IP 
core using hardware simulation tools and deployed onto a target PYNQ-Z2 board. Interfacing via a software-managed hardware overlay, 
this pipeline is made to maximize throughput and achieve low cycle latency, delivering an entirely independent, standalone edge 
acceleration system for disease detection and classification. 
