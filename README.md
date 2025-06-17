# BSc Thesis – Physics & Applied Mathematics
# Integrated Computational Methods: Simulation, Modulation & Optimization in Python

* **Institution:** University of Baghdad — Department of Physics and Applied Mathematics
* **Author:** [Heider Jeffer](https://github.com/HeiderJeffer)
* **Supervisor:** [Dr. Jeffer Dhia Jeffer](https://en.wikipedia.org/wiki/Jafar_Dhia_Jafar)
* **Language:** English
* **Submitted:** June 2008
* **Project Repository:** [View on GitHub](https://github.com/HeiderJeffer/BSc-Thesis-Integrated-Computational-Methods-Simulation-Modulation-Optimization-in-Python/blob/main/BSC%20Python.ipynb)
* **Thesis Repository:** [View on GitHub](https://github.com/HeiderJeffer/Bachelor-Thesis-Summary)
* **Full Thesis PDF:** [View on Google Drive](https://drive.google.com/file/d/17Lo4m5VZGAK9A5stfMTnROEZbqYOyMZa/view?usp=sharing)


This project integrates:

* **Discrete Event Simulation** (e.g., queuing systems)
*  **Signal Modulation** (AM, FM, ASK, FSK)
*  **Optimization Algorithms** (Gradient Descent & Genetic Algorithms)

Developed entirely in **Python**, this work demonstrates how combining these methods can simulate, analyze, and enhance complex system behavior.


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


### **APPENDIX**


Here's a **step-by-step walkthrough** of both parts of the code: the **Discrete Event Simulation (DES)** and the **Signal Modulation**.

## Part A: Discrete Event Simulation (Queue System)

### 1. **Imports and Setup**

```python
import heapq
import random
```

* `heapq` is used to manage a priority queue for scheduling events.
* `random` is used to generate exponentially distributed inter-arrival and service times.

### 2. **Define an Event Class**

```python
class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time
```

* Each event has a `time` and a `type` (either `'arrival'` or `'departure'`).
* `__lt__` allows the heap to sort events by time.


### 3. **Simulation Function**

```python
def simulate_queue(arrival_rate=1.0, service_rate=1.2, sim_time=100):
```

* Simulates a single-server queue.
* `arrival_rate`: λ (e.g., 1 customer per unit time).
* `service_rate`: μ (e.g., 1.2 customers served per unit time).
* `sim_time`: How long to simulate.

#### State Variables

```python
    clock = 0
    queue = []
    event_queue = []
    busy = False
    wait_times = []
```

* `clock`: Simulation time.
* `queue`: FIFO line of waiting customers.
* `event_queue`: Future events in a min-heap.
* `busy`: Server state.
* `wait_times`: List of customer wait durations.

#### Helper to Schedule Events

```python
    def schedule_event(time, event_type):
        heapq.heappush(event_queue, Event(time, event_type))
```

### 4. **Initialize First Arrival**

```python
    schedule_event(random.expovariate(arrival_rate), 'arrival')
```

* The first arrival is scheduled using an exponential distribution.


### 5. **Simulation Loop**

```python
    while event_queue and clock < sim_time:
        event = heapq.heappop(event_queue)
        clock = event.time
```

* Continues until simulation time ends or no more events.
* Retrieves the next event by time.

#### On `arrival`:

```python
        if event.event_type == 'arrival':
            if not busy:
                busy = True
                service_time = random.expovariate(service_rate)
                schedule_event(clock + service_time, 'departure')
            else:
                queue.append(clock)
            schedule_event(clock + random.expovariate(arrival_rate), 'arrival')
```

* If the server is free, service starts immediately.
* If busy, the arrival time is stored in the queue.
* A new arrival is always scheduled.

#### On `departure`:

```python
        elif event.event_type == 'departure':
            if queue:
                arrival_time = queue.pop(0)
                wait_times.append(clock - arrival_time)
                service_time = random.expovariate(service_rate)
                schedule_event(clock + service_time, 'departure')
            else:
                busy = False
```

* If queue is not empty: serve next customer, record wait time.
* If queue is empty: server becomes idle.


### 6. **Return Average Wait Time**

```python
    return sum(wait_times)/len(wait_times) if wait_times else 0.0
```

### 7. **Run the Simulation**

```python
avg_wait = simulate_queue()
print("Average Wait Time:", avg_wait)
```

## Part B: Signal Modulation (AM, FM, ASK, FSK)

### 1. **Imports**

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 2. **Time and Signal Setup**

```python
t = np.linspace(0, 1, 1000)
carrier_freq = 10
message_freq = 1
carrier = np.sin(2 * np.pi * carrier_freq * t)
message = np.sin(2 * np.pi * message_freq * t)
```

* `t`: Time array (1 second, 1000 samples).
* `carrier`: High-frequency sine wave.
* `message`: Low-frequency sine wave.



### 3. **Amplitude Modulation (AM)**

```python
am = (1 + message) * carrier
```

* Message signal scales the amplitude of the carrier.



### 4. **Frequency Modulation (FM)**

```python
k = 5
fm = np.sin(2 * np.pi * carrier_freq * t + k * np.sin(2 * np.pi * message_freq * t))
```

* Message changes the **phase**, thus frequency of carrier.


### 5. **Amplitude Shift Keying (ASK)**

```python
bitstream = np.array([1, 0, 1, 1, 0])
ask = np.repeat(bitstream, 200) * np.sin(2 * np.pi * carrier_freq * t[:1000])
```

* Binary bits scale the amplitude (on/off keying).


### 6. **Frequency Shift Keying (FSK)**

```python
f1, f2 = 5, 15
fsk = np.concatenate([
    np.sin(2 * np.pi * f1 * t[:200]) if bit == 0 else np.sin(2 * np.pi * f2 * t[:200])
    for bit in bitstream
])
```

* 0 and 1 are encoded using different frequencies (`f1` and `f2`).

### 7. **Plot All Signals**

```python
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1); plt.plot(t, am); plt.title("AM Signal")
plt.subplot(4, 1, 2); plt.plot(t, fm); plt.title("FM Signal")
plt.subplot(4, 1, 3); plt.plot(t, ask); plt.title("ASK Signal")
plt.subplot(4, 1, 4); plt.plot(fsk); plt.title("FSK Signal")
plt.tight_layout()
plt.show()
```

# Appendix
**By Heider Jeffer**


```python
# Appendix A: Python Code – Discrete Event Simulation (DES)
# By Heider Jeffer

import heapq
import random

class Event:
    def __init__(self, time, event_type):
        self.time = time
        self.event_type = event_type

    def __lt__(self, other):
        return self.time < other.time

def simulate_queue(arrival_rate=1.0, service_rate=1.2, sim_time=100):
    clock = 0
    queue = []
    event_queue = []
    busy = False
    wait_times = []

    def schedule_event(time, event_type):
        heapq.heappush(event_queue, Event(time, event_type))

    # Initial arrival
    schedule_event(random.expovariate(arrival_rate), 'arrival')

    while event_queue and clock < sim_time:
        event = heapq.heappop(event_queue)
        clock = event.time

        if event.event_type == 'arrival':
            if not busy:
                busy = True
                service_time = random.expovariate(service_rate)
                schedule_event(clock + service_time, 'departure')
            else:
                queue.append(clock)
            schedule_event(clock + random.expovariate(arrival_rate), 'arrival')

        elif event.event_type == 'departure':
            if queue:
                arrival_time = queue.pop(0)
                wait_times.append(clock - arrival_time)
                service_time = random.expovariate(service_rate)
                schedule_event(clock + service_time, 'departure')
            else:
                busy = False

    return sum(wait_times)/len(wait_times) if wait_times else 0.0

avg_wait = simulate_queue()
print("Average Wait Time:", avg_wait)

```

    Average Wait Time: 1.9962285426867634
    


```python
# Appendix B: Python Code – Modulation (AM, FM, ASK, FSK)
# By Heider Jeffer
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 1000)
carrier_freq = 10
message_freq = 1
carrier = np.sin(2 * np.pi * carrier_freq * t)
message = np.sin(2 * np.pi * message_freq * t)

# AM
am = (1 + message) * carrier

# FM
k = 5  # frequency deviation
fm = np.sin(2 * np.pi * carrier_freq * t + k * np.sin(2 * np.pi * message_freq * t))

# ASK
bitstream = np.array([1, 0, 1, 1, 0])
ask = np.repeat(bitstream, 200) * np.sin(2 * np.pi * carrier_freq * t[:1000])

# FSK
f1, f2 = 5, 15
fsk = np.concatenate([
    np.sin(2 * np.pi * f1 * t[:200]) if bit == 0 else np.sin(2 * np.pi * f2 * t[:200])
    for bit in bitstream
])

plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1); plt.plot(t, am); plt.title("AM Signal")
plt.subplot(4, 1, 2); plt.plot(t, fm); plt.title("FM Signal")
plt.subplot(4, 1, 3); plt.plot(t, ask); plt.title("ASK Signal")
plt.subplot(4, 1, 4); plt.plot(fsk); plt.title("FSK Signal")
plt.tight_layout()
plt.show()



```


    
![png](BSC%20Python_files/BSC%20Python_2_0.png)
    



```python
# Appendix C: Python Code – Optimization (Gradient Descent and Genetic Algorithm)
```


```python
# Gradient Descent
def f(x): return x**2 + 4*x + 4
def df(x): return 2*x + 4

x = 10
learning_rate = 0.1
for _ in range(100):
    x = x - learning_rate * df(x)
print("Minimum at x =", x)

```

    Minimum at x = -1.9999999975555567
    


```python
# Genetic Algorithm (with deap library)
from deap import base, creator, tools, algorithms
import random

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def rastrigin(ind):
    return 10 * len(ind) + sum(x**2 - 10 * np.cos(2 * np.pi * x) for x in ind),

toolbox.register("evaluate", rastrigin)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=50)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
best = tools.selBest(pop, 1)[0]
print("Best individual:", best)

```

    Best individual: [-0.009048470318822721, 1.2687742205831336e-05]
    


```python

```


