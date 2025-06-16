## **Bachelor’s Thesis Summary**

**Institution**: University of Baghdad Department of Physics and Apply Mathematics	

**Title**: Integrated Computational Methods: Simulation, Modulation & Optimization in Python

**Author**: Heider Jeffer	

**Supervisor**: Dr. Jafar Dhia Jafar	

**Language**: English

**Date**: June 2008 





### **Introduction and Aim**

This Bachelor’s thesis investigates the integration of three core computational approaches—**simulation**, **modulation**, and **optimization**—as applied through the Python programming language. The study stems from the recognition that many real-world systems are too complex for closed-form solutions, and computational tools must be leveraged to model, predict, and improve performance.


The primary aim is to develop a framework in Python that enables simultaneous exploration of these methods, demonstrating their individual and combined effectiveness. Applications are drawn from engineering, signal processing, and operations research, thereby grounding the work in practical relevance.



### **Theoretical Background**

#### **Simulation (Discrete Event Simulation)**

Discrete Event Simulation (DES) is a modeling technique where system changes occur at discrete points in time. This method is widely used for understanding queuing systems, logistics networks, and service processes. DES captures complex time-dependent behaviors and is well-suited for stochastic modeling. It involves defining events (e.g., customer arrivals), system states, and rules for time advancement.

#### **Modulation (Analog & Digital)**

Modulation techniques are used in communications to encode and transmit information. The thesis investigates both analog (AM, FM) and digital (ASK, FSK) schemes. Theoretical definitions and time-domain representations are covered. For instance, Amplitude Modulation (AM) encodes data in the amplitude of a carrier wave, whereas Frequency Modulation (FM) varies the carrier frequency. Digital methods shift between discrete states and are typically more robust in noisy environments.

#### **Optimization (Gradient Descent & Genetic Algorithms)**

Optimization seeks the best solution from a set of feasible alternatives. Two algorithmic strategies are compared:

* **Gradient Descent (GD)**: A deterministic technique suited for convex problems.
* **Genetic Algorithms (GA)**: Stochastic, population-based heuristics that are useful for exploring complex, multimodal landscapes.

These tools are assessed for convergence behavior, adaptability, and robustness.



### **Research Questions**

1. How can discrete event simulation be implemented in Python to represent queuing dynamics?
2. What are the differences in behavior between analog and digital modulation techniques under varying noise levels?
3. How do optimization algorithms like GD and GA compare in solving nonlinear objective functions?
4. Can these methods be integrated to enhance system-level performance?



### **Methodology**

#### **Tools and Technologies**

* **Language**: Python 3.10
* **Libraries**: NumPy, Matplotlib, Heapq, Scikit-opt
* **Hardware**: Intel Core i5, 16GB RAM

#### **Simulation Module**

A DES model simulates a two-server queuing system where arrivals follow a Poisson process and service times are exponentially distributed. Events include arrival, service start, and service completion. A min-heap (`heapq`) structure manages time-ordered events. Performance metrics such as average waiting time, server utilization, and queue length are calculated over time.

#### **Modulation Module**

Sinusoidal waveforms are used to model AM and FM, with digital equivalents created for ASK and FSK. Time-series graphs are plotted using Matplotlib, and noise effects are modeled with additive Gaussian noise. The demodulation and decoding accuracy under noisy conditions are also analyzed.

#### **Optimization Module**

Two functions are targeted:

* A **simple quadratic function** for gradient descent.
* The **Rastrigin function**, known for multiple local minima, for genetic algorithms.

GA performance is measured by fitness score evolution, mutation/crossover success, and convergence stability.

#### **Integrated Case Study**

The final integration optimizes the service rate parameter in the DES model using GA to minimize average wait time. This demonstrates real-time application of optimization to a dynamic simulation problem.



### **Results and Findings**

#### **Simulation Findings**

* Queue performance is sensitive to the load factor (λ/μ).
* High utilization leads to exponential increases in waiting times.
* Visual graphs show how customer flow and wait times evolve over time.

#### **Modulation Findings**

* **FM** performs better than **AM** in noisy environments due to constant amplitude.
* **ASK** is more error-prone under noise than **FSK**.
* Graphs of waveforms confirm expected theoretical patterns, validating the simulation.

#### **Optimization Outcomes**

* **Gradient Descent** converges quickly but fails in functions with multiple minima.
* **Genetic Algorithms** are more resilient and effective in rugged landscapes, albeit slower.
* GA was successfully applied to tune the DES model, showing measurable improvements in system efficiency.


### **Discussion**

This project confirms that integrating simulation, modulation, and optimization can generate deeper insights than using any one method alone. Simulation allows exploration of dynamic systems, modulation enables encoding/decoding of complex signals, and optimization ensures these systems operate at peak performance.

Python proves to be a robust platform for this integration due to its flexibility, package ecosystem, and visualization capabilities. The thesis reflects a strong emphasis on applied problem-solving and the synthesis of theory and practice.



### **Conclusion**

The study successfully demonstrates the integrated application of simulation, modulation, and optimization within a single Python-based analytical framework. Each module supports the others—simulation generates data, modulation encodes it, and optimization improves outcomes. This framework can be expanded for real-world use in fields such as telecommunications, supply chain analysis, and smart system design.



### **Future Directions**

* Incorporate **reinforcement learning** for adaptive optimization.
* Extend DES to model **stochastic networks** with multiple decision nodes.
* Integrate **real-time sensor data** for dynamic modulation and system control.



### **References**

* Banks, J. et al. (1996). *Discrete-event system simulation*. Prentice Hall.
* Proakis, J. G. (2001). *Digital Communications*. McGraw-Hill.
* Yang, R. (1994). "Genetic Algorithms and Their Applications," *Artificial Intelligence*.
* Nocedal, J., & Wright, S. J. (1999). *Numerical Optimization*. Springer.
* Gross, D. & Harris, C. M. (1998). *Fundamentals of Queueing Theory*. Wiley.



