# Illustration of the Dual Branch PQP CNN 
This directory depicts the entire flow of dual branch architecture with ECG input taken as input and the output bmp files obtained from the RTl simulation.


## 1. Illustration of Convolution operation for ECG input image
This is demonstrated by the sample calculation shown in the figure:
<img width="883" height="479" alt="image" src="https://github.com/user-attachments/assets/9d10e2cc-7c4a-4864-97b3-3a8b3d34738e" />


## 2. ECG image representation of individual branch outputs
The dual-branch architecture generates two feature maps from separate convolution paths, as shown in figure 
<img width="976" height="480" alt="image" src="https://github.com/user-attachments/assets/d2809351-68cd-4dcd-929e-7760fec2833a" />


The second-layer output of Braanch 1 contains relatively faint waveform details, whereas the output of Branch 2 shows more prominent ECG peaks and waveform variations. These outputs are added element by element to produce the adder output, which retains the prominent peaks while incorporating contributions from both branches.

## 3. Representation of ECG signal before and after Max Pooling Unit
<img width="976" height="480" alt="image" src="https://github.com/user-attachments/assets/d1127731-38e0-4629-9071-3bae2fceafb9" />
After max-pooling, the output size is reduced to 254x254 while the major waveform structures remain visible. For each non-overlapping 2x2 window, the maximum of the four input values is selected, reducing the spatial dimensions by half along both axes.

## 4.Implementational flow of Dual branch CNN architecture

<img width="1013" height="375" alt="image" src="https://github.com/user-attachments/assets/894cc6b5-b1cf-4e33-940c-b27715d493ff" />

The prominent ECG peaks remain visible throughout these stages, indicating that the main waveform structures are retained after convolution, addition, and down sampling.



