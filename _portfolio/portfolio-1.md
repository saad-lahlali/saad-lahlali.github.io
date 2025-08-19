---
title: "Teaching a Robot to Follow You with Reinforcement Learning"
excerpt: "How do you teach a robot to follow a person without programming its every move? You let it learn. This article chronicles the journey of Teresa, a simulated robot we taught to autonomously track a person using only its camera and the power of reinforcement learning. We break down everything from the core AI principles to the open-source tools you'll need, showing how a simple reward system can create intelligent behavior and providing a roadmap to building your own autonomous agent. <br/><img src='/images/portfolio-1/saad_robot.png' style='display: block; margin-left: auto; margin-right: auto; width: 50%;'>"
collection: portfolio
---


*How we trained a Gazebo-simulated robot named Teresa to autonomously follow a person, and how you can do it too.*

What if a robot could learn to follow a person simply by looking through a camera? This isn't science fiction; it's the reality of what we can achieve with the power of reinforcement learning. This portfolio chronicles the journey of Teresa, a simulated robot, from a stationary object to an autonomous agent capable of tracking and following a person.

This portfolio was born out of a desire to create an intelligent, autonomous system that could interact with its environment in a meaningful way. We'll walk you through the entire process, from the underlying reinforcement learning principles to the nitty-gritty of the code and the tools we used. By the end, you'll not only understand how Teresa works but also have a roadmap to build something similar yourself.

-----

## The Brains of the Operation: Reinforcement Learning 🧠

At the heart of this project is a powerful machine learning paradigm: **Reinforcement Learning (RL)**. Imagine teaching a dog a new trick. You reward it for good behavior and don't for bad. RL works in a similar way.

<br/><img src='/images/portfolio-1/rl_map.png'>



Here's how we applied it to Teresa:

  - **The Agent**: The robot, Teresa.
  - **The Environment**: A 3D world created using **Gazebo**, a popular robotics simulator. This world contains Teresa, a ground plane, and a model of a standing person.
  - **The State**: The "state" is what the robot perceives from the environment. In our case, the state is derived from the camera feed. We use person detection (either Haarcascades or Tiny Yolo) to determine if the person is centered in the robot's view. The robot's goal is to learn from this visual input, which is processed into an 800-dimensional vector.
  - **The Actions**: Teresa has a simple set of four possible movements: rotate right, rotate left, move backward, and move forward.
  - **The Reward**: This is the crucial part. The robot receives a positive reward when the person is centered in its camera view. This positive feedback encourages the robot to learn the actions that lead to this outcome.

The goal of the training process is for the robot to learn a **policy**—a strategy that tells it which action to take in any given state to maximize its total reward.

<br/><img src='/images/portfolio-1/what_robot_sees.png'>

-----

## The Tools of the Trade 🛠️

To bring Teresa to life, we relied on a suite of powerful, open-source tools:

  - **ROS (Robot Operating System)**: The backbone of our project. ROS provides a flexible framework for writing robot software. We used the Melodic version.
  - **Gazebo**: A robust 3D robotics simulator that integrates seamlessly with ROS. This allowed us to create a virtual world for Teresa to train in, complete with realistic physics.
  - **`roslibpy`**: A Python library that enabled our training script to communicate with the ROS environment. This is how we sent movement commands to the robot and received data from its sensors.
  - **TensorFlow**: The deep learning framework we used to build and train our reinforcement learning model.

-----

## A Deeper Dive: How the Model Learns 🤖

The magic that allows Teresa to learn is a reinforcement learning algorithm called **Monte Carlo Policy Gradient**, also known as **REINFORCE**. The goal is simple: adjust the robot's decision-making process, or "policy," to maximize the total reward it receives over time. To handle the complex environment, we use a deep neural network to represent this policy.

#### The Optimization Goal

The core idea is to perform **gradient ascent**, a process of iteratively adjusting the policy's parameters to find the best possible strategy. The update rule for the policy's parameters, represented by the Greek letter theta ($θ$), looks like this:

$$θ ← θ + α∇_θJ(θ)$$

Let's break that down:

  - $$θ$$ represents the parameters of our policy network—specifically, the weights and biases of the neural network.
  - $$α$$ is the **learning rate**, which controls how big of a step we take during each update. In this project, it's set to 0.01.
  - $$∇_θJ(θ)$$ is the **gradient** of the expected reward. Think of it as an arrow pointing in the direction that will increase our reward the most.

To figure out which direction that "arrow" should point, the algorithm estimates the gradient using the experience gathered from a full episode:

$$∇_θJ(θ) ≈ Σ_{t=0}^T ∇_θ\log{π_θ(a_t|s_t)}G_t$$

This formula is the heart of the algorithm:
  - $$π_θ(a_t\|s_t)$$ is the **policy**. It’s the probability of the robot taking a specific action ($a\_t$) when in a particular state ($s\_t$).
  - $$∇_θ\log{π_θ(a_t\|s_t)}$$ is a vector that points in the direction to make the action we actually took ($a\_t$) more likely in the future.
  - $$G_t$$ is the **discounted cumulative future reward**. It’s a number that tells us how "good" the outcome was from that point ($t$) until the end of the episode, with a discount factor `gamma` of `0.95` applied to future rewards.

**The intuition is powerful**: the formula multiplies how *good* an action's outcome was ($G\_t$) by the direction needed to make that action more likely ($$∇\_θ\\log{π}$$). If an action led to a high reward, the policy is changed to make that action more probable in the future. If it led to a poor outcome, its probability is decreased.

#### The Role of the Neural Network

A deep neural network is the perfect tool to represent the policy ($π\_θ$) because it can learn complex behaviors from the high-dimensional data coming from the robot's camera.

Here's how it works in our `train.py` script:

  - **Function Approximation**: The neural network acts as a **function approximator**. It takes the environment's state as input (`input`) and, after passing it through several hidden layers (`fc1`, `fc2`, `fc3`), outputs a probability distribution over all possible actions (`action_distribution`) using a softmax function. This output *is* the policy.
  - **Connecting to the Code**: Instead of manually programming the gradient ascent, we use TensorFlow's automatic differentiation. We define a `loss` function, and by minimizing it, we are effectively maximizing our reward. The loss function in the code directly mirrors the policy gradient formula:

<!-- end list -->

```python
loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_)
```

  - `neg_log_prob` corresponds to $$-\log{π_θ(a_t\|s_t)}$$ from our formula.
  - `discounted_episode_rewards_` corresponds to the future reward, $$G_t$$.

By asking the `AdamOptimizer` to minimize this loss, we are telling TensorFlow to automatically calculate the gradients and update the network's weights ($$θ$$) in a way that makes action sequences with high rewards more likely.

-----

## The Results: An Autonomous Follower 🏆

After training, Teresa is able to successfully follow the person in the simulation. The saved model, `saved_model.ckpt`, can be loaded to test its performance.

Here is the robot during its training phase, learning through trial and error.

And here is the final result, where the trained robot successfully tracks and follows the person.

The beauty of using a standardized platform like ROS is that this same trained model can be deployed on the real Teresa robot with minimal changes, bridging the gap between simulation and the real world. You can see a full video of the project [here](https://www.youtube.com/watch?v=Qo_Pitp4Zk8&ab_channel=SaadLahlali).

<p align="center">
  <img src="/images/portfolio-1/train_gif.gif" alt="Train">
</p>


<p align="center">
  <img src="/images/portfolio-1/test_gif.gif" alt="Test">
</p>

-----

## What's Next? 🚀

This project lays a solid foundation for more advanced autonomous robot behaviors. The next steps could involve:

  - **Obstacle Avoidance**: Integrating additional sensors like LiDAR to enable the robot to navigate more complex environments and avoid obstacles.
  - **More Complex Behaviors**: Training the robot to perform more sophisticated tasks, such as finding a specific person in a room.
  - **Real-World Testing**: Deploying the trained model on the physical Teresa robot and evaluating its performance in a real-world setting.

-----

## Code ✅

You can access the full source code at the [Teresa_Robot GitHub Repository](https://github.com/saad2050l/Teresa_Robot/tree/master).