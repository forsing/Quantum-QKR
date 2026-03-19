"""
QKR - Quantum Kernel Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def compute_quantum_kernel():
    n_states = 1 << NUM_QUBITS
    fmap = ZZFeatureMap(feature_dimension=NUM_QUBITS, reps=1)

    statevectors = []
    for v in range(n_states):
        feat = value_to_features(v)
        circ = fmap.assign_parameters(feat)
        sv = Statevector.from_instruction(circ)
        statevectors.append(sv)

    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            fid = abs(statevectors[i].inner(statevectors[j])) ** 2
            K[i, j] = fid
            K[j, i] = fid

    return K


def kernel_regression(K, y, lam=LAMBDA_REG):
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * np.eye(n), y)
    pred = K @ alpha
    pred = np.maximum(pred, 0)
    if pred.sum() > 0:
        pred /= pred.sum()
    return pred


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, prob in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- Kvantni kernel (ZZFeatureMap, {NUM_QUBITS}q, reps=1) ---")
    K = compute_quantum_kernel()
    print(f"  Kernel matrica: {K.shape}, rang: {np.linalg.matrix_rank(K)}")

    print(f"\n--- QKR po pozicijama (lambda={LAMBDA_REG}) ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = kernel_regression(K, y)
        dists.append(pred)
        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QKR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Kvantni kernel (ZZFeatureMap, 5q, reps=1) ---
  Kernel matrica: (32, 32), rang: 32

--- QKR po pozicijama (lambda=0.01) ---
  Poz 1 [1-33]: 1:0.168 | 2:0.146 | 3:0.130
  Poz 2 [2-34]: 8:0.086 | 5:0.076 | 9:0.075
  Poz 3 [3-35]: 13:0.064 | 12:0.063 | 14:0.061
  Poz 4 [4-36]: 23:0.063 | 21:0.062 | 18:0.062
  Poz 5 [5-37]: 29:0.065 | 26:0.063 | 27:0.063
  Poz 6 [6-38]: 33:0.083 | 32:0.081 | 35:0.080
  Poz 7 [7-39]: 7:0.182 | 38:0.152 | 37:0.132

==================================================
Predikcija (QKR, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""




"""
QKR - Quantum Kernel Regression

ZZFeatureMap (5 qubita, reps=1) enkodira svaku mogucu vrednost (0-31) u kvantno stanje. 
Kvantni kernel: fidelity K(i,j) = |<phi(i)|phi(j)>|^2 izmedju svih 32 stanja (32x32 matrica). 
Kernel ridge regresija: gladi empirijsku distribuciju kroz kvantni kernel. 
Egzaktne verovatnoce preko Statevector - potpuno deterministicki. 
Greedy selekcija za finalnu kombinaciju. 
Brze je od QCBM jer nema iterativno treniranje - kernel se racuna jednom.
"""
