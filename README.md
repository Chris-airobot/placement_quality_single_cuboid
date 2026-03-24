# Placement Quality for a Single Cuboid

A simulation-driven robotics project for predicting grasp-conditioned placement quality on a single cuboid object.

This repository explores how grasp choice affects downstream placement feasibility and quality, using physics-based simulation, motion validation, and learning-based scoring to support more efficient robot decision-making.

## System diagram

![System diagram](assets/system_diagram.png)

**System overview.** Offline object and grasp poses are sampled in simulation without physics, path-wise IK and collision labels are generated, and a dual-head MLP is trained to predict IK and collision feasibility. The learned scores are then evaluated under physics-enabled execution using motion planning and large-scale executed-simulation rollouts.

## Overview

In many pick-and-place systems, grasping and placing are treated as separate problems. This project studies their coupling by asking a more specific question:

**Given a candidate grasp and a target placement pose, can the robot place the object successfully and reliably?**

The repository focuses on a single cuboid setting and uses simulation to generate labeled data for learning grasp-conditioned placement feasibility and quality.

## Project goals

- Generate candidate grasps and target placements in simulation
- Evaluate placement feasibility under kinematic and collision constraints
- Collect large-scale labeled data for grasp–place pairs
- Train predictive models to estimate placement quality
- Support rank-then-plan execution by filtering or prioritizing promising candidates

## Main idea

Instead of planning every grasp–placement pair from scratch at runtime, this project builds a predictive model from simulation-generated data.

The repository is organized around one **final successful pipeline** and several **earlier experimental iterations**.

The final pipeline has three main stages:

1. **Offline data generation**  
   Candidate grasps, initial poses, and target placement poses are sampled in simulation without physics. Each grasp–placement pair is labeled using path-wise inverse-kinematics and collision checks.

2. **Model training**  
   A dual-head MLP is trained to predict:
   - IK feasibility
   - collision feasibility

3. **Executed-simulation evaluation**  
   The learned scores are used to rank grasp–placement candidates, which are then tested under physics-enabled execution with motion planning.

## Repository structure

```text
.
├── assets/                         # Figures, diagrams, and demo images
├── docker/                         # Docker build and environment setup files
├── docs/                           # Setup notes, troubleshooting, and legacy notes
├── final_pipeline/
│   └── 04_path_simulation/         # Final successful path-aware validation pipeline
├── legacy_experiments/             # Earlier research iterations kept for reference
│   ├── 01_cube_simulation/            # First cuboid simulation pipeline
│   ├── 02_ycb_simulation/             # Intermediate object-set simulation experiments
│   └── 03_cube_generalization/        # Third generalization attempt
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.sh