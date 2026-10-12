# 12435611_topology-optimization

This repository contains the project elaborate in the course "Applied Deep Learning" from the Technical University of Vienna.

This README is also the Assignment 1 deliverable.

## Project Description and Type

The goal of this project is to create a deep learning model for topology optimization.

**Project Type:** "Bring your own method" (potentially even "beat the stars")

## Topology Optimization and the Role of Deep Learning

Very broadly speaking, topology optimization is a method commonly used in engineering to create the optimal shape of a structure given certain loads and constraints. 
A typical example for this (which is also what will be addressed in this project) is a mechanical component, that should be designed to be as lightweight as possible while still being able to withstand the applied loads. I.e. the goal might be to minimize the amount of material used, maintaining a certain minimal stiffness. Or the other way around, maximizing stiffness (in the field of topology optimization this is often referred to as minimizing compliance), given a certain volume of material.

While a lot of topology optimization types and methods exist, the basic idea is usually the same, namely iterative improvement based on some kind of computation or simulation. An example could be to start with an initial design, or fully filled volume. Then perform finite element analysis (FEA) or some other type of simulation is to compute the property of interest, e.g. mechanical stress (von Mises stress) within the structure. Given the results, the structure is modified. E.g. in low stress regions, material is removed. After that a new FEA is performed and this process is repeated iteratively until convergence. Other methods use gradient-based optimization or heuristic algorithms to guide the material distribution, but all require repeated and often costly evaluations of each iteration.

In summary, classical topology optimization methods iteratively improve the design based on the results of FEA. However, these can be extremely computationally expensive. Additionally, given the iterative nature of the process, even efficient simulations can amount to a long time throughout the whole optimization process. 

Deep learning models may offer highly efficient alternative, as they may skip the whole iterative optimization process and directly predict the optimal design given the loads and constraints.

## Literature and Approach

Various approaches have been proposed for using deep learning in topology optimization, e.g. Convolutional NN [1], GANs [2] or Diffusion Models [3]. A diffusion based foundation model has also been published in [7]

My approach in this project will be to implement a transformer based model for directly predicting the optimal design from the inputs. This idea has not been explored extensively in the literature so far. In fact, I found very little prior work, mainly from a single group of researchers at The University of North Carolina at Charlotte. They have published two papers on this topic, one being the evolution of the previous one. The articles are [4] and [5]. However, they dont reveal details on their specific architecture and hyperparamenter. A framework for transformers on images (Vision Transformer, ViT) has already been introduced by Dosovitskiy et al. in [6] which has been the basis for the transformers for topology optimization in the aforementioned papers. Depending on the exact inputs I may follow this framework while building my own architecture. 

The exact input format and information is not completely clear yet and will be part of the exploration. However, there will definitely be information about the loads and boundary conditions as well as volume fraction of the final result. I.e. how much of the total available volume will be occupied by material. The input might be enhanced e.g. by e.g. stress distribution from an initial FEA simulation. Material properties will not be included as I will assume isotropic and homogeneous material for simplicity and absolute stress values will be normalized if necessary. 


## Data Set

My goal is to compare my results to the results in papers [4] and [5]. Therefore, I have contacted the authors to request access to their data set for a fair comparison. However, since a cannot rely on this I created my own script (see: `generate_data.py`) for generating a data set using the topology optimization library [topoptlab](https://github.com/stefanhiemer/topoptlab), from which all necessary data can be obtained.


## Setup

Python 3.12.4 or newer is required by `topoptlab`. With [uv](https://docs.astral.sh/uv/) installed, set up the environment using:

```bash
uv sync
```

## Citations

Articles referenced in this README:

- **[1]** Sosnovik, I. and Oseledets, I. (2018). *[Neural Networks for Topology Optimization](
https://doi.org/10.48550/arXiv.1709.09578)*.

- **[2]** Zhenguo Nie et al. (2020). *[TopologyGAN: Topology Optimization using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain](https://doi.org/10.48550/arXiv.2003.04685)*.

- **[3]** Francois Mazé and Faez Ahmed (2022). *[Diffusion Models Beat GANs on Topology Optimization](https://doi.org/10.48550/arXiv.2208.09591)*.

- **[4]** Aaron Lutheran, Srijan Das and Alireza Tabarraei (Sept. 2025). *[Tansformer-based Topology Optimization](https://doi.org/10.48550/arXiv.2509.05800)*.

- **[5]** Aaron Lutheran, Srijan Das and Alireza Tabarraei (May 2026). *[Physics-informed transformer for real-time high-fidelity topology optimization](https://doi.org/10.1007/s00158-026-04342-6)*.

- **[6]** Dosovitskiy et al. (Oct. 2020). *[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://doi.org/10.48550/arXiv.2010.11929)*.

- **[7]** Amin Heyrani Nobari, Lyle Regenwetter, Cyril Picard, Ligong Han and Faez Ahmed (2025). *[Optimize Any Topology: A Foundation Model for Shape- and Resolution-Free Structural Topology Optimization](https://doi.org/10.48550/arXiv.2510.23667)*.
