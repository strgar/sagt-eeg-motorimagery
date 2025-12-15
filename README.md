# SAGT-EEG Motor Imagery Classification

This repository provides the official implementation of the proposed
Self-Adaptive Graph Transformer (SAGT) for EEG Motor Imagery classification.

## Method Overview
The proposed framework consists of:
- Wavelet-based feature extraction using Daubechies-5 (db5)
- Phase Locking Value (PLV) for functional connectivity
- Gaussian kernel smoothing
- Local Adaptive Connectivity Prior (LACP)
- Self-Adaptive Graph Transformer (SAGT)

## Input Representation
Each EEG trial is represented as:
- Node features: Mu and Beta band energies
- Graph edges: PLV-based adjacency refined with Gaussian kernel and LACP

## Dataset
The dataset is not included in this repository due to licensing constraints.
Experiments were conducted using PhysioNet EEG Motor Movement/Imagery dataset
and Emotiv Epoch+ recordings.

## Usage
This repository focuses on method implementation.
Users are expected to provide their own EEG dataset.

## Citation
If you use this code, please cite the corresponding research paper.
