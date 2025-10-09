# QUBO and Ising Models for Quantum Annealing - Quick Reference Guide

## Overview

This guide provides a practical reference for encoding optimization problems as QUBO (Quadratic Unconstrained Binary Optimization) or Ising models for quantum annealing applications.

---

## 1. Basic Formulations

### QUBO (Quadratic Unconstrained Binary Optimization)

**Objective:** Minimize $f(x) = x^T Q x$ where $x_i \in \{0, 1\}$

**Expanded form:**
$$f(x) = \sum_i Q_{ii} x_i + \sum_{i<j} Q_{ij} x_i x_j$$

**Properties:**
- Binary decision variables
- Quadratic objective function
- No explicit constraints (encoded via penalties)
- $Q$ is typically upper triangular

### Ising Model

**Classical Hamiltonian:**
$$H_{\text{Ising}} = \sum_{i<j} J_{ij} s_i s_j + \sum_i h_i s_i$$

where $s_i \in \{-1, +1\}$

**Quantum Hamiltonian:**
$$\hat{H}_{\text{Ising}} = \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z + \sum_i h_i \sigma_i^z$$

**Parameters:**
- $J_{ij}$: Coupling between spins $i$ and $j$
  - $J_{ij} > 0$: Ferromagnetic (spins align)
  - $J_{ij} < 0$: Antiferromagnetic (spins anti-align)
- $h_i$: Local magnetic field (bias) on spin $i$

---

## 2. Conversion Between QUBO and Ising

### QUBO → Ising

**Variable transformation:** $s_i = 1 - 2x_i$

**Result:** $x_i = \frac{1 - s_i}{2}$

**Products:** $x_i x_j = \frac{(1-s_i)(1-s_j)}{4} = \frac{1 - s_i - s_j + s_i s_j}{4}$

**Algorithm:**
```python
def qubo_to_ising(Q):
    """Convert QUBO matrix Q to Ising parameters J, h"""
    n = len(Q)
    J = {}
    h = np.zeros(n)
    offset = 0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i] += -0.5 * Q[i,i]
                offset += 0.5 * Q[i,i]
            elif i < j:
                J[(i,j)] = 0.25 * Q[i,j]
                h[i] += 0.25 * Q[i,j]
                h[j] += 0.25 * Q[i,j]
                offset += 0.25 * Q[i,j]
    
    return J, h, offset
```

### Ising → QUBO

**Variable transformation:** $x_i = \frac{1 - s_i}{2}$

**Result:** $s_i = 1 - 2x_i$

**Products:** $s_i s_j = (1-2x_i)(1-2x_j) = 1 - 2x_i - 2x_j + 4x_i x_j$

---

## 3. Common Problem Encodings

### Max-Cut Problem

**Problem:** Partition graph vertices to maximize edges between partitions

**QUBO formulation:**
$$\min f(x) = -\sum_{(i,j) \in E} (x_i + x_j - 2x_i x_j)$$

**Ising formulation:**
$$\min H(s) = \sum_{(i,j) \in E} s_i s_j$$

**Interpretation:**
- Edges within same partition: $x_i = x_j \Rightarrow x_i x_j = x_i = x_j$ (penalty)
- Edges across partition: $x_i \neq x_j \Rightarrow x_i x_j = 0$ (no penalty)

### Number Partitioning

**Problem:** Partition numbers $\{a_1, ..., a_n\}$ into two sets with equal sums

**QUBO formulation:**
$$\min f(x) = \left(\sum_i a_i x_i - \frac{S}{2}\right)^2$$

where $S = \sum_i a_i$

**Expanded:**
$$f(x) = \sum_i (a_i^2 - a_i S) x_i + \sum_{i<j} 2a_i a_j x_i x_j + \text{const}$$

### Graph Coloring (k colors)

**Constraint:** No adjacent vertices have same color

**Encoding:** Use $k$ binary variables per vertex: $x_{i,c}$ = 1 if vertex $i$ has color $c$

**Constraints as penalties:**
1. Each vertex has exactly one color: $\sum_c x_{i,c} = 1$
   - Penalty: $A \left(\sum_c x_{i,c} - 1\right)^2$

2. Adjacent vertices have different colors: $x_{i,c} \cdot x_{j,c} = 0$ for all $(i,j) \in E$
   - Penalty: $B \sum_{(i,j) \in E} \sum_c x_{i,c} x_{j,c}$

**Total QUBO:**
$$f(x) = A \sum_i \left(\sum_c x_{i,c} - 1\right)^2 + B \sum_{(i,j) \in E} \sum_c x_{i,c} x_{j,c}$$

---

## 4. Quantum Annealing Schedule

### Standard Hamiltonian

$$\hat{H}(t) = A(t) \hat{H}_{\text{driver}} + B(t) \hat{H}_{\text{problem}}$$

**Components:**
- $\hat{H}_{\text{driver}} = -\sum_i \sigma_i^x$ (transverse field)
- $\hat{H}_{\text{problem}}$ = Ising Hamiltonian encoding the problem
- $A(t)$, $B(t)$ are time-dependent coefficients

### Typical Schedule (Linear)

$$A(t) = A_0 \left(1 - \frac{t}{T}\right)$$
$$B(t) = B_0 \frac{t}{T}$$

**Properties:**
- At $t=0$: $A(0) = A_0$, $B(0) = 0$ → Ground state is $\ket{+}^{\otimes N}$
- At $t=T$: $A(T) = 0$, $B(T) = B_0$ → Ground state encodes solution

### Adiabatic Condition

For successful quantum annealing:
$$T \gg \frac{\hbar}{\Delta_{\min}^2}$$

where $\Delta_{\min}$ is the minimum energy gap during evolution.

**Practical implication:** Annealing time must be long enough relative to the smallest gap.

---

## 5. Hardware Considerations

### D-Wave Quantum Annealers

**Chimera Graph (older generation):**
- Unit cells: $K_{4,4}$ bipartite graphs
- Connectivity: ~6 neighbors per qubit
- Total qubits: 2000+

**Pegasus Graph (newer generation):**
- Higher connectivity: ~15 neighbors per qubit
- Total qubits: 5000+
- Better for dense problems

### Minor Embedding

**Challenge:** Problem graph rarely matches hardware graph directly

**Solution:** Use multiple physical qubits to represent one logical qubit

**Process:**
1. Find embedding: Map logical variables to chains of physical qubits
2. Add strong ferromagnetic couplings within chains: $J_{\text{chain}} \ll -1$
3. Map problem couplings to inter-chain connections

**Trade-offs:**
- Uses more physical qubits
- Requires strong chain couplings (limited by hardware)
- Chain breaks = errors in solution

**Tools:**
- `minorminer` (automatic embedding)
- D-Wave Ocean SDK

---

## 6. Practical Tips

### Choosing Penalty Weights

When encoding constraints as penalties:

1. **Make penalties strong enough:**
   - Penalty weight $A$ should be larger than the largest problem coefficient
   - Rule of thumb: $A \geq 2 \times \max(|Q_{ij}|)$

2. **Don't make penalties too strong:**
   - Very large penalties can cause precision issues
   - May create very small energy gaps
   - Typical range: $A \in [1, 100]$ relative to problem scale

### Scaling the Problem

**Normalize coefficients:**
```python
def normalize_qubo(Q):
    """Normalize QUBO to [-1, 1] range"""
    max_abs = np.max(np.abs(Q))
    return Q / max_abs, max_abs
```

### Interpreting Results

**From quantum annealer:**
- Multiple samples returned
- Each sample may violate soft constraints
- Need to:
  1. Check constraint satisfaction
  2. Evaluate objective function
  3. Keep best valid solution

**Post-processing:**
- Local search refinement
- Constraint repair
- Ensemble methods (combine multiple runs)

---

## 7. Example Code

### Complete Max-Cut Implementation

```python
import numpy as np
import networkx as nx

class MaxCutQUBO:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph.nodes())
    
    def build_qubo(self):
        """Build QUBO matrix for max-cut"""
        Q = np.zeros((self.n, self.n))
        
        for i, j in self.graph.edges():
            Q[i, i] += -1
            Q[j, j] += -1
            if i < j:
                Q[i, j] += 2
            else:
                Q[j, i] += 2
        
        return Q
    
    def to_ising(self):
        """Convert to Ising parameters"""
        J = {}
        h = np.zeros(self.n)
        
        for i, j in self.graph.edges():
            J[(min(i,j), max(i,j))] = 1.0
        
        return J, h
    
    def evaluate(self, x):
        """Evaluate solution quality"""
        Q = self.build_qubo()
        return x @ Q @ x
    
    def get_cut_size(self, partition):
        """Count edges in the cut"""
        cut = 0
        for i, j in self.graph.edges():
            if partition[i] != partition[j]:
                cut += 1
        return cut

# Example usage
G = nx.erdos_renyi_graph(10, 0.5, seed=42)
maxcut = MaxCutQUBO(G)
Q = maxcut.build_qubo()
J, h = maxcut.to_ising()

# Solve classically (brute force for small problems)
best_energy = float('inf')
best_x = None

for i in range(2**maxcut.n):
    x = np.array([(i >> j) & 1 for j in range(maxcut.n)])
    energy = maxcut.evaluate(x)
    if energy < best_energy:
        best_energy = energy
        best_x = x

print(f"Best cut size: {maxcut.get_cut_size(best_x)}")
print(f"Partition: {best_x}")
```

---

## 8. Common Pitfalls and Solutions

### Pitfall 1: Incorrect QUBO Symmetry
**Problem:** QUBO matrix should be upper triangular
**Solution:** Ensure $Q[i,j] = 0$ for $i > j$, or symmetrize: $Q \leftarrow (Q + Q^T)/2$

### Pitfall 2: Constraint Violations
**Problem:** Penalty weights too small
**Solution:** Increase penalty weights; validate on small examples

### Pitfall 3: Poor Embedding
**Problem:** Not all problem couplings can be embedded
**Solution:** Simplify problem or use different hardware graph

### Pitfall 4: Chain Breaks
**Problem:** Physical qubits in same logical chain disagree
**Solution:** 
- Increase chain coupling strength
- Use majority voting
- Run multiple times and filter broken chains

### Pitfall 5: Precision Issues
**Problem:** Very large or very small coefficients
**Solution:** Normalize problem to $[-1, 1]$ or similar range

---

## 9. Resources

### Software Tools
- **D-Wave Ocean SDK:** Full quantum annealing toolkit
- **PyQUBO:** Pythonic QUBO formulation
- **dimod:** Reference implementations for samplers
- **NetworkX:** Graph manipulation and analysis

### Further Reading
- Glover et al. (2019): "A Tutorial on Formulating and Using QUBO Models"
- Lucas (2014): "Ising formulations of many NP problems"
- D-Wave Documentation: Problem formulation guides

### Benchmarks
- **QAOA circuits:** Compare with gate-based approaches
- **Simulated annealing:** Classical baseline
- **Gurobi/CPLEX:** Exact solvers for small instances

---

## Summary

**Key Principles:**
1. QUBO and Ising are equivalent - choose based on convenience
2. Constraints become penalties in the objective
3. Quantum annealing success depends on energy gap structure
4. Hardware embedding is non-trivial for dense problems
5. Multiple runs and post-processing often necessary

**When Quantum Annealing Shines:**
- Problems with natural QUBO/Ising structure
- Large solution spaces benefiting from quantum sampling
- Applications where near-optimal solutions are valuable
- Problems with favorable gap structure

**Current Limitations:**
- Exponentially small gaps for many NP-hard problems
- Hardware connectivity constraints
- Limited to specific problem types
- Debate over quantum advantage

The field is rapidly evolving - stay updated with latest research and hardware developments!
