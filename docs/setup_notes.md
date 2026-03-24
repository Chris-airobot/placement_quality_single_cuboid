# Grasping for placement project
## Project High-Level Pipeline
1. **Generate Candidate Grasps:** Use Dex-Net or another grasp planner.
2. **For Each Grasp, Simulate Placements:** Randomize placement strategies and record outcomes.
3. **Collect Data & Outcomes:** Store object descriptors (weight, geometry via TSDF), and placement results (pose shifts, success/failure, etc.). <span style="color:blue"> Maybe we can also find the enegry consumption?</span>.
4. **Aggregate & Model a Distribution:** From multiple placement attempts per grasp, estimate a distribution over outcomes.
5. **Train a Predictive Model:** Input: (object representation, target pose, grasp parameters) → Output: expected placement quality.
6. **Runtime Use:** For a new object, run grasp generation and use the trained model to select the best grasp.

A few notes:
- Maybe use some common shapes that could represent most objects
- Idea of the second step,:
    - After grasping the object, define a range of final poses where the object might be placed
    - For each of the final pose, use some common planner to place down the object, the strategy could be dropping the object from random distance, or directly touching the ground
- Idea of the forth step:
    - $Q = S_{success} ∗ (w_1 ∗ D + w_2 ∗ T)$ where $T$ is time taken, and $D$ is the pose shift
- For Sim-to-real gap, two potential methods:
    - **Domain randomization:** randomly vary fricition, angle of the ground, and add noises to sensors
    - **Real data fine-tuning:** After the model has been trained in simulation, also collects a small dataset in reality to fine-tune the model

## Discussions:
1. Approaches for grasping and placement should be the same for training and testing?
 - Yes. But it feels like a bit limited considering there are infinite ways to grasping and placing
2. Steps:
    - Random samples the object -> Use grasp planner to generate a bunch of grasps -> Place it down for a specific goal pose 
    - Input of the model: initial pose of the object, grasp pose, and the final pose of the object
    - Start with fixing the orientation of the object, simply varying the positions

3. When generating the goal pose, some poses may be not be feasible (like a donut is vertically placed). One way is to randomly drop the object to find discrete feasible poses.

4. Consider the absolute poses?



## Simulator Choices:
The simulator plan to use is IsaacSim

## Robot Connection 
The robot plan to use is UFactory xarm 7, for linux system, set up the wire connection with:
- Address: 192.168.1.12
- Netmask: 255.255.255.0
- Gateway: 192.168.1.1
- DNS: 192.168.1.1

For the robot in the lab, IP address is **192.168.1.209**, i.e. Type "192.168.1.209:18333" to use UFactory Studio



