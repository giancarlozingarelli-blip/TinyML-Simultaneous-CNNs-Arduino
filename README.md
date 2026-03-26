# EdgeAI: Simultaneous CNNs on ARM Cortex-M4

This repository contains the work developed for my Master's Thesis in Electronic Engineering at **Politecnico di Bari**. The project focuses on the simultaneous deployment of two Deep Learning models on a resource-constrained embedded system.

## 🚀 Project Overview
The core objective is the concurrent execution of two Convolutional Neural Networks (CNNs) on a single **Arduino Nano 33 BLE Sense** microcontroller.

1. **Keyword Spotting (KWS):** A voice recognition model designed for smart home applications.
2. **Obstacle Detection:** A vision/distance-based model for autonomous driving systems.

## 🛠️ Technical Specifications
* **Hardware:** Arduino Nano 33 BLE Sense (ARM Cortex-M4, 256KB RAM, 1MB Flash).
* **Framework:** TensorFlow Lite Micro.
* **Optimization:** Post-training quantization (INT8) to meet memory, latency, and power consumption constraints.
* **Languages/Tools:** Python (Keras/TensorFlow for training), C++ (Firmware/Arduino IDE).

## 📈 Key Results
The models were optimized to ensure smooth real-time inference, achieving an optimal balance between accuracy and energy efficiency on edge hardware.

---
*Author: Giancarlo Zingarelli*
