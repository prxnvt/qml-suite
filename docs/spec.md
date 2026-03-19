# QML Suite — Product Requirements & Technical Specification

**Version:** 0.1 (Ideation / Pre-implementation)
**Date:** 2026-03-19
**Source reference:** `~/workspace/qbit-weaver/` (React/TypeScript browser app, 8-qubit cap)

---

## 0. Purpose of This Document

This spec describes a native C++ desktop application that:
1. Extends qbit-weaver's quantum circuit simulator to 16 qubits
2. Adds a full Quantum Machine Learning (QML) module with differentiable circuit execution and training loops
3. Provides a rich visualization suite suited to research and education
4. Exposes a Python API via pybind11 for scripted QML workflows

Every section is written so that a future implementer can derive the exact behavior without inspecting qbit-weaver. Where qbit-weaver behavior is reused verbatim (e.g. gate matrices), the exact formulas are reproduced here.

---

## 1. Glossary

| Term | Meaning |
|------|---------|
| **qubit** | Two-level quantum system; indexed 0…n-1, row 0 = most significant bit (MSB) |
| **state vector** | Complex array of 2^n amplitudes; amplitude at index `i` is `α_i ∈ ℂ` |
| **basis state** | An index `i ∈ [0, 2^n)` corresponding to the computational basis element `|i⟩` |
| **gate** | Unitary transformation applied to one or more qubits |
| **column** | One time step in the circuit grid; all gates in a column execute simultaneously on the state before the column |
| **PQC** | Parameterized Quantum Circuit — a circuit with trainable real-valued parameters θ |
| **QML** | Quantum Machine Learning — using PQCs as trainable models |
| **parameter shift rule** | Exact gradient formula: ∂⟨O⟩/∂θ = [⟨O⟩(θ+π/2) − ⟨O⟩(θ−π/2)] / 2 |
| **adjoint differentiation** | Reverse-mode gradient computation through the circuit; 1 forward + 1 backward pass for all parameters (§6.4) |
| **expectation value** | ⟨O⟩ = ⟨ψ|O|ψ⟩ for observable O and state |ψ⟩ |
| **ansatz** | A fixed circuit architecture template whose parameters are varied during training |
| **barren plateau** | Phenomenon where gradient variance vanishes exponentially with qubit count in deep PQCs (§6.10) |
| **shot** | One run of a quantum circuit followed by a measurement collapse |
| **SIMD** | Single Instruction Multiple Data — AVX2/AVX-512 vector operations |
| **sparse state** | State represented as a sorted vector of nonzero (basis-state index, amplitude) pairs |
| **dense state** | State represented as a contiguous `2^n`-element complex array |
| **CSD** | Cosine-Sine Decomposition — recursive factorization of a unitary into single/two-qubit gates (used in §6.8 QPCA) |
| **QPE** | Quantum Phase Estimation — subroutine that estimates eigenvalues of a unitary operator |

---

## 2. High-Level Architecture

```
qml_suite/
├── CMakeLists.txt                  # Top-level CMake
├── core/
│   ├── engine/                     # State vector simulation kernel
│   ├── gates/                      # Gate matrix library
│   ├── circuit/                    # Circuit model, history, serializer
│   └── measurement/                # Born-rule measurement, shot sampling
├── qml/
│   ├── pqc/                        # Parameterized circuits
│   ├── gradients/                  # Parameter shift, adjoint diff
│   ├── optimizers/                 # Adam, SGD, SPSA, COBYLA
│   ├── encodings/                  # Data → quantum state
│   ├── kernels/                    # Quantum kernel matrices
│   └── algorithms/                 # VQE, QAOA, QNN, QSVM, QPCA
├── ui/
│   ├── MainWindow/
│   ├── CircuitEditor/
│   ├── GateLibrary/
│   ├── Visualization/
│   └── QMLWorkspace/
├── python/
│   └── bindings/                   # pybind11 → Python package `qml_suite`
└── tests/
    ├── engine/
    └── qml/
```

The application is built as a **single executable** (`qml_suite`) linked against Qt6 for the GUI. The simulation kernel (`core/`) is compiled as a static library so it can also be linked into the pybind11 Python module without the Qt dependency.

---

## 3. Simulation Engine (`core/engine/`)

### 3.1 State Vector Representation

#### Dense format

The primary state vector storage is an `alignas(64) std::vector<double>` with interleaved real/imaginary parts, identical to qbit-weaver's `Float64Array` layout:

```
data[2*i]     = Re(α_i)
data[2*i + 1] = Im(α_i)
```

For n qubits, `data.size() == 2 * 2^n`. The array must be 64-byte aligned so AVX-512 loads/stores are valid on any element boundary.

**Type alias:**
```cpp
using StateVec = std::vector<double>;  // length = 2 * 2^n, 64-byte aligned
```

Use `std::aligned_alloc(64, 2 * (1<<n) * sizeof(double))` or a custom allocator for the backing storage.

Accessor macros (defined once, used everywhere):
```cpp
inline double  re(const StateVec& s, int i)              { return s[2*i];   }
inline double  im(const StateVec& s, int i)              { return s[2*i+1]; }
inline void    set(StateVec& s, int i, double r, double x){ s[2*i]=r; s[2*i+1]=x; }
inline void    add(StateVec& s, int i, double r, double x){ s[2*i]+=r; s[2*i+1]+=x; }
inline bool    isZero(const StateVec& s, int i, double eps=1e-10){
    return std::abs(s[2*i]) < eps && std::abs(s[2*i+1]) < eps;
}
```

#### Sparse format

When the fraction of nonzero amplitudes drops below **10%** AND `n >= 12`, use a sparse hash map:

```cpp
struct SparseState {
    int numQubits;
    std::vector<std::pair<int, std::complex<double>>> entries; // sorted by basis-state index, ascending
};
```

Maintain the sort invariant: after a full gate application pass that produces a new entries vector, sort once via `std::sort` at the end rather than maintaining order incrementally during accumulation. Use `std::lower_bound` for any point lookups (e.g., control mask checks). See §17 open question 2 for rationale.

The threshold logic: after each column is applied, count nonzero amplitudes. If `nonzeroCount / 2^n < 0.10` and `n >= 12`, convert to sparse. If it exceeds 50%, convert back to dense. Conversions must be exact (round-trip lossless except for EPSILON pruning at `1e-10`).

**Thrashing prevention:** after any sparse↔dense conversion, skip format re-evaluation for the next `sparseConversionCooldown` columns (default: 3, configurable in `settings.json` under `simulation.sparseConversionCooldown`). This prevents pathological circuits (e.g., alternating H gates that spread and collapse the state) from converting back and forth every column. Additionally, the user/config may pin the format to always-dense or always-sparse via `simulation.forceStateFormat: "dense" | "sparse" | "auto"` (default `"auto"`).

**Initial state** always starts as `|0⟩`: dense array of zeros with `data[0] = 1.0` (Re), `data[1] = 0.0` (Im).

### 3.2 Qubit Ordering Convention

**Row 0 = MSB (most significant bit).** Qubit at row `r` maps to bit position:

```
bit_position(r) = numQubits - 1 - r
```

Basis state index `i` has qubit `r` in state `|b⟩` iff bit `bit_position(r)` of `i` equals `b`. This is identical to qbit-weaver.

### 3.3 Single-Qubit Gate Application

Given a 2×2 unitary `M` and target row `r`, the new state is:

```
For each basis state index i:
  bit = bit_position(r)
  localIdx = (i >> bit) & 1          // qubit value: 0 or 1
  otherBits = i & ~(1 << bit)        // all other bits unchanged

  For k in {0, 1}:
    targetIdx = otherBits | (k << bit)
    newState[targetIdx] += M[k][localIdx] * state[i]
```

**Zero-skipping:** if `isZero(state, i)`, skip entirely. This is O(nonzero) rather than O(2^n).

**Parallelism:** Iterate over `i` values in two interleaved groups: those with `bit=0` and `bit=1`. Each group is independent and can be parallelized with OpenMP. The parallel threshold should be `n >= 10` (1024 amplitudes), below which thread overhead exceeds benefit.

**Control mask:** Before applying the matrix, check:
```
if ((i & controlMask) != controlMask) { newState[i] += state[i]; continue; }
if ((i & antiControlMask) != 0)       { newState[i] += state[i]; continue; }
```
where `controlMask` has a 1 at each bit position of a CONTROL qubit and `antiControlMask` has a 1 at each ANTI_CONTROL qubit. Both are computed once per column from the gate's control/anti-control annotations (see §4.4).

### 3.4 SWAP Gate Application

```
For each pair (i, j) where j = i ^ (1<<bit1) ^ (1<<bit2) and i < j:
  Check controlMask and antiControlMask on i
  If qubit values at bit1 and bit2 differ: swap amplitudes at i and j
```

This avoids creating a second array; operate in-place by only swapping when `i < j`.

### 3.5 Multi-Qubit Spanning Gate Application (QFT)

QFT on qubit rows `[startRow, endRow]` (inclusive) of span size `s = endRow - startRow + 1`:

Build the full `2^s × 2^s` DFT matrix:
```
QFT[j][k] = (1/√(2^s)) × exp(2πi·j·k / 2^s)
```

For the inverse (QFT†), negate the exponent:
```
QFTDG[j][k] = (1/√(2^s)) × exp(-2πi·j·k / 2^s)
```

Application: for each "non-span context" (the `2^(n-s)` combinations of all non-span qubits), extract the `2^s` amplitudes corresponding to this context, multiply by the DFT matrix, and write back. This is `O(2^n · 2^s)` for a naive matrix-vector multiply; for large spans use a Cooley-Tukey FFT decomposition instead (`O(n · 2^n)`).

**Index reconstruction:** Given a non-span context integer `ctx` and a span value `spanVal`:
- Iterate qubit rows 0…n-1
- If row is in [startRow, endRow]: bit from `spanVal` at position `(row - startRow)`, ordered MSB-first within the span (i.e. bit `s-1-(row-startRow)`)
- Otherwise: bit from `ctx` at the appropriate position among non-span qubits (ordered MSB-first among non-span qubits)

### 3.6 Arithmetic Permutation Gates

Arithmetic gates act as **permutation operators** on the basis: they map each basis state `|i⟩` to `|f(i)⟩` where `f` is a bijection. Implementation:

```
newState = zeros(2^n)
For each i:
  if isZero(state, i): continue
  check controlMask, antiControlMask
  effectVal = readRegister(i, effectStart, effectEnd)
  inputA    = readRegister(i, inputAStart, inputAEnd)   // if register assigned
  inputB    = readRegister(i, inputBStart, inputBEnd)   // if register assigned
  inputR    = readRegister(i, inputRStart, inputREnd)   // if register assigned
  newVal    = computeArithmetic(gateType, effectVal, inputA, inputB, inputR, mod2n)
  j         = writeRegister(i, newVal, effectStart, effectEnd)
  newState[j] += state[i]
```

**`readRegister(basisState, startRow, endRow)`:** Little-endian within the span. The **top qubit** (lowest row index = startRow) is the **LSB**:
```
value = 0
for i in 0..(spanSize-1):
  row = startRow + i
  bit = bit_position(row)
  if (basisState >> bit) & 1: value |= (1 << i)
return value
```

**`writeRegister(basisState, value, startRow, endRow)`:** Clear span bits, set from value using the same convention.

**`mod2n = 1 << (endRow - startRow + 1)`.**

**`computeArithmetic` truth table** (all arithmetic is modular unless _MOD_R variant, in which case R is the modulus and the operation is a no-op when effectVal >= R):

| GateType | Output |
|----------|--------|
| INC | (effectVal + 1) % mod2n |
| DEC | (effectVal − 1 + mod2n) % mod2n |
| ADD_A | (effectVal + inputA) % mod2n |
| SUB_A | (effectVal − inputA + mod2n) % mod2n |
| MUL_A | (effectVal × inputA) % mod2n — only if inputA is odd (otherwise identity) |
| DIV_A | (effectVal × modInverse(inputA, mod2n)) % mod2n — only if inputA is odd |
| MUL_B | same as MUL_A but uses inputB |
| DIV_B | same as DIV_A but uses inputB |
| INC_MOD_R | if effectVal < R: (effectVal+1) % R, else identity |
| DEC_MOD_R | if effectVal < R: (effectVal−1+R) % R, else identity |
| ADD_A_MOD_R | if effectVal < R and inputA < R: (effectVal+inputA) % R, else identity |
| SUB_A_MOD_R | if effectVal < R and inputA < R: (effectVal−inputA+R) % R, else identity |
| MUL_A_MOD_R | if effectVal < R and gcd(inputA,R)==1: (effectVal×inputA) % R, else identity |
| DIV_A_MOD_R | if effectVal < R and gcd(inputA,R)==1: (effectVal×modInverse(inputA,R)) % R |

**Comparison gates** (A_LT_B, A_LEQ_B, A_GT_B, A_GEQ_B, A_EQ_B, A_NEQ_B): flip target qubit if the comparison between inputA and inputB is true, otherwise identity. Target qubit is the cell's row.

**Scalar gates** (SCALE_I, SCALE_NEG_I, SCALE_SQRT_I, SCALE_SQRT_NEG_I): multiply every controlled amplitude by a global scalar — `i`, `-i`, `e^(iπ/4)`, `e^(-iπ/4)` respectively.

### 3.7 Input-Parameterized Gate Application (ZA, XA, YA, ZB, XB, YB)

These gates read an integer value from an input register and use it as a rotation angle. The convention: input value `k` from a span of size `s` corresponds to angle `θ = 2π·k / 2^s`.

**ZA** (Z^A): apply phase `e^(iθ)` to `|1⟩` component of target qubit (leaves `|0⟩` unchanged):
```
For each i where bit at target is 1:
  k = readRegister(i, inputAStart, inputAEnd)
  θ = 2π·k / 2^(inputASize)
  amplitude[i] *= exp(iθ)
```

**XA** (X^A): apply `Rx(θ)` to target qubit:
```
Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
```
For each pair `(i0, i1)` where `i0` has target qubit = 0 and `i1 = i0 ^ targetBitMask`:
- Read inputA from `i0` (the `|0⟩` partner)
- Compute `θ = 2π·k / 2^s`, `c = cos(θ/2)`, `s_val = sin(θ/2)`
- `newAmp[i0] = c × amp[i0] + s_val × Im(amp[i1]) + i×(c × Im(amp[i0]) - s_val × Re(amp[i1]))`
  Wait — exact formula from quantum.ts:
  ```
  result[i0].re = c * amp0Re + s * amp1Im
  result[i0].im = c * amp0Im - s * amp1Re
  result[i1].re = s * amp0Im + c * amp1Re
  result[i1].im = -s * amp0Re + c * amp1Im
  ```

**YA** (Y^A): apply `Ry(θ)` to target qubit:
```
Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
```
```
result[i0].re = c * amp0Re - s * amp1Re
result[i0].im = c * amp0Im - s * amp1Im
result[i1].re = s * amp0Re + c * amp1Re
result[i1].im = s * amp0Im + c * amp1Im
```

**ZB, XB, YB**: identical to ZA/XA/YA but read from the InputB register span.

### 3.8 Phase Gradient Gate

Applied to a span `[startRow, endRow]` of size `s`:
```
For each basis state i:
  k = readRegister(i, startRow, endRow)
  θ = 2π·k / 2^s
  amplitude[i] *= exp(iθ)
```

### 3.9 Time-Parameterized Gates

A global time parameter `t ∈ [0.0, 1.0]` is animated (default update rate: 60 fps, step ≈ 0.005 per frame, wrapping at 1.0). Gate matrices:

| Gate | Matrix |
|------|--------|
| ZT (Z^t) | `[[1, 0], [0, exp(2πit)]]` |
| XT (X^t) | `[[cos(πt), -i·sin(πt)], [-i·sin(πt), cos(πt)]]` |
| YT (Y^t) | `[[cos(πt), -sin(πt)], [sin(πt), cos(πt)]]` |
| EXP_Z | `[[exp(iπt), 0], [0, exp(-iπt)]]` |
| EXP_X | `[[cos(πt), i·sin(πt)], [i·sin(πt), cos(πt)]]` |
| EXP_Y | `[[cos(πt), sin(πt)], [-sin(πt), cos(πt)]]` |

These gates cannot be cached; their matrices are recomputed for each simulation tick that uses them.

### 3.10 Measurement

**Single-qubit measurement (Born rule):**
```
prob_0 = Σ |α_i|² for all i where bit_position(r) of i = 0
r ~ Bernoulli(prob_0)    // draw random number ∈ [0,1)
result = 0 if r < prob_0 else 1
```

**Post-measurement state collapse:**
```
For each i:
  if bit at position bit_position(qubit) of i ≠ result: α_i = 0
  else: α_i /= √(probability_of_result)
```

MEASURE gates in the circuit are processed **after** all other gates in the same column complete. Multiple measurements in the same column are applied left-to-right, top-to-bottom (sorted by row), each collapsing the state before the next measurement.

**Shot-based sampling:** Run the full circuit `shots` times from `|0⟩`, each with a fresh RNG state. Aggregate result counts. Used by the QML module for expectation value estimation.

### 3.11 Bloch Vector Extraction

For target qubit `k` in an n-qubit state, the Bloch vector `(X, Y, Z)` is:

```
X = 2 × Σ Re(conj(α_i0) × α_i1)     for all pairs (i0, i1) where i1 = i0 | targetBitMask and i0 has target bit = 0
Y = 2 × Σ Im(conj(α_i1) × α_i0)     (same pairs)
Z = Σ |α_i0|² - Σ |α_i1|²           (prob of 0 minus prob of 1)
```

Clamp each coordinate to `[-1, 1]` and snap values with `|v| < 1e-10` to exactly 0.

Exact formula from source (reproduced):
```cpp
for each i where (i >> bitK) & 1 == 0:
  i0 = i, i1 = i | (1 << bitK)
  // accumulate
  expX += 2 * (Re(state[i0]) * Re(state[i1]) + Im(state[i0]) * Im(state[i1]))
  expY += 2 * (Im(state[i1]) * Re(state[i0]) - Re(state[i1]) * Im(state[i0]))
  expZ += |state[i0]|² - |state[i1]|²
```

### 3.12 Performance Requirements

| Scenario | Requirement |
|----------|-------------|
| 16-qubit single gate application | < 5 ms on modern desktop (≥8 logical cores) |
| 16-qubit full 20-column circuit simulation | < 500 ms |
| 12-qubit circuit with sparse state (density 1%) | < 10 ms per gate |
| Bloch vector extraction (16 qubits) | < 2 ms |
| QFT on 16 qubits | < 100 ms |

These drive the requirement for OpenMP parallelism and SIMD. The gate loop inner body (complex multiply-add) must be vectorizable: avoid branch-heavy code on the hot path; precompute control masks outside the loop.

---

## 4. Gate Library (`core/gates/`)

### 4.1 Fixed 2×2 Gate Matrices

All matrix entries given as `(re, im)` pairs. `inv2 = 1/√2 ≈ 0.7071067811865476`.

| GateType | Matrix (row-major) |
|----------|--------------------|
| X (Pauli-X) | `[[0,0), (1,0)], [(1,0), (0,0)]]` |
| Y (Pauli-Y) | `[[(0,0), (0,-1)], [(0,1), (0,0)]]` |
| Z (Pauli-Z) | `[[(1,0), (0,0)], [(0,0), (-1,0)]]` |
| H (Hadamard) | `[[(inv2,0), (inv2,0)], [(inv2,0), (-inv2,0)]]` |
| S | `[[(1,0), (0,0)], [(0,0), (0,1)]]` |
| SDG (S†) | `[[(1,0), (0,0)], [(0,0), (0,-1)]]` |
| T | `[[(1,0), (0,0)], [(0,0), (inv2,inv2)]]` |
| TDG (T†) | `[[(1,0), (0,0)], [(0,0), (inv2,-inv2)]]` |
| I (Identity) | `[[(1,0), (0,0)], [(0,0), (1,0)]]` |
| SQRT_X (√X) | `[[(0.5,0.5), (0.5,-0.5)], [(0.5,-0.5), (0.5,0.5)]]` — equivalent to Rx(π/2) |
| SQRT_X_DG | Rx(-π/2) |
| SQRT_Y | Ry(π/2) |
| SQRT_Y_DG | Ry(-π/2) |

Computed rotation gates (all take angle `θ` in radians):
```
Rx(θ) = [[cos(θ/2),        (0,-sin(θ/2))],
          [(0,-sin(θ/2)),   cos(θ/2)     ]]

Ry(θ) = [[cos(θ/2),  -sin(θ/2)],
          [sin(θ/2),   cos(θ/2)]]

Rz(θ) = [[exp(-iθ/2),  0        ],
          [0,            exp(iθ/2)]]
       = [[(cos(-θ/2), sin(-θ/2)), (0,0)],
          [(0,0),                  (cos(θ/2), sin(θ/2))]]
```

Preset angle gates (cached on startup):
- RX_PI_2, RX_PI_4, RX_PI_8, RX_PI_12 → Rx(π/2), Rx(π/4), Rx(π/8), Rx(π/12)
- RY_PI_2, RY_PI_4, RY_PI_8, RY_PI_12 → Ry(π/2), ..., Ry(π/12)
- RZ_PI_2, RZ_PI_4, RZ_PI_8, RZ_PI_12 → Rz(π/2), ..., Rz(π/12)

### 4.2 Gate Caching Policy

- **Fixed gates** (X, Y, Z, H, S, T, etc.): precompute at startup, store as `std::array<std::complex<double>, 4>` (row-major 2×2), never recompute.
- **Preset rotation gates** (RX_PI_2, …): precompute at startup, cache by enum value.
- **Parameterized gates** (RX, RY, RZ with user-supplied angle): compute on each `applyGate` call with the given `angle` from `GateParams`. Do NOT cache (angle can be different per cell).
- **Time-parameterized gates** (ZT, XT, YT, EXP_Z, EXP_X, EXP_Y): compute per simulation tick with current `t`. Do NOT cache.
- **Custom gates**: stored inline in `GateParams::customMatrix`. Use directly, no caching.

### 4.3 Full Gate Enum

```cpp
enum class GateType {
    // Pauli
    X, Y, Z,
    // Clifford
    H, S, SDG, T, TDG, I,
    // Square root
    SQRT_X, SQRT_X_DG, SQRT_Y, SQRT_Y_DG,
    // Parameterized rotation
    RX, RY, RZ,
    // Preset rotations
    RX_PI_2, RX_PI_4, RX_PI_8, RX_PI_12,
    RY_PI_2, RY_PI_4, RY_PI_8, RY_PI_12,
    RZ_PI_2, RZ_PI_4, RZ_PI_8, RZ_PI_12,
    // Multi-qubit
    CX, CY, CZ, SWAP, CCX,
    // Control markers (not applied as gates; decode to controlMask)
    CONTROL, ANTI_CONTROL, X_CONTROL, X_ANTI_CONTROL, Y_CONTROL, Y_ANTI_CONTROL,
    // Time-parameterized
    ZT, XT, YT,
    // Input-parameterized
    ZA, XA, YA, ZB, XB, YB,
    // Exponential
    EXP_Z, EXP_X, EXP_Y,
    // QFT
    QFT, QFT_DG,
    // Arithmetic — effect register
    INC, DEC,
    ADD_A, SUB_A, MUL_A, DIV_A,
    MUL_B, DIV_B,
    INC_MOD_R, DEC_MOD_R,
    ADD_A_MOD_R, SUB_A_MOD_R, MUL_A_MOD_R, DIV_A_MOD_R,
    // Arithmetic — comparison (target qubit flipped if true)
    A_LT_B, A_LEQ_B, A_GT_B, A_GEQ_B, A_EQ_B, A_NEQ_B,
    // Scalar phase
    SCALE_I, SCALE_NEG_I, SCALE_SQRT_I, SCALE_SQRT_NEG_I,
    // Input register markers (visual only, decoded by column parser)
    INPUT_A, INPUT_B, INPUT_R,
    // Spanning / special
    PHASE_GRADIENT, REVERSE,
    // Visualization-only
    BLOCH_VIS, PERCENT_VIS,
    // User
    MEASURE, CUSTOM,
    NONE,  // empty cell
};
```

### 4.4 Control Marker Semantics

Control markers appear in the same circuit column as the gate they modify. The column parser scans for them before applying gates:

- **CONTROL**: the row is a Z-basis control. Qubit must be `|1⟩` for gate to fire. Adds a 1 to `controlMask` at `bit_position(row)`.
- **ANTI_CONTROL**: qubit must be `|0⟩`. Adds a 1 to `antiControlMask` at `bit_position(row)`.
- **X_CONTROL / X_ANTI_CONTROL**: X-basis control. Apply `H` to this qubit before the gate and `H` again after. Post-basis-change, X_CONTROL acts like ANTI_CONTROL and X_ANTI_CONTROL like CONTROL (this is the exact behavior from qbit-weaver).
- **Y_CONTROL / Y_ANTI_CONTROL**: Y-basis control. Apply `H·S†` before and `S·H` after. Post-basis-change, Y_CONTROL → ANTI_CONTROL, Y_ANTI_CONTROL → CONTROL.

The column simulator therefore:
1. Collects all X_CONTROL, X_ANTI_CONTROL, Y_CONTROL, Y_ANTI_CONTROL rows
2. Applies H (or H·S†) to those qubits
3. Applies the target gate with the combined controlMask/antiControlMask
4. Re-applies H (or S·H) to those qubits

### 4.5 REVERSE Gate

The REVERSE gate spans multiple rows (startRow…endRow) and reverses the bit ordering within that span. It is implemented as a sequence of `⌊spanSize/2⌋` SWAP operations.

### 4.6 Angle Expression Parser

Accepts string expressions for gate angles. Supported syntax:
- Numeric literals: `0.5`, `-3.14`
- `pi` (case-insensitive) → `M_PI`
- Arithmetic: `+`, `-`, `*`, `/`
- Functions: `sqrt(x)`, `abs(x)`, `sin(x)`, `cos(x)`, `exp(x)`
- Parentheses
- Examples: `"pi/4"`, `"3*pi/8"`, `"sqrt(2)/2"`, `"-pi/12"`

Returns `double` or an error with a human-readable message indicating the position of the parse failure.

---

## 5. Circuit Model (`core/circuit/`)

### 5.1 Data Structures

```cpp
struct GateParams {
    double angle = 0.0;                              // For RX, RY, RZ
    std::string angleExpression;                     // User input e.g. "pi/4"
    std::optional<std::array<std::complex<double>,4>> customMatrix; // For CUSTOM (row-major 2×2)
    std::string customLabel;                         // Short label for CUSTOM gate display
    struct Span { int startRow; int endRow; };
    std::optional<Span> spanOverride;               // For REVERSE, QFT, PHASE_GRADIENT
    bool isSpanContinuation = false;                 // True for cells below span head
};

struct Cell {
    GateType gate = GateType::NONE;
    std::string id;           // UUID, generated on insertion
    GateParams params;
};

// CircuitGrid: rows × cols, row-major
using CircuitGrid = std::vector<std::vector<Cell>>;
```

**Grid constraints:**
- Minimum 1 row, maximum **16 rows** (enforced in UI and simulation)
- Minimum 1 column; UI starts with 10 columns and auto-expands when the last column has a gate
- Maximum 64 columns (hard cap to prevent runaway memory from history)
- Default initial state: 8 rows × 10 columns, all NONE

### 5.2 Column Analysis

Before simulating a column, the column analyzer resolves the complete gate configuration:

1. **Identify control markers:** scan all rows for CONTROL, ANTI_CONTROL, X_CONTROL, X_ANTI_CONTROL, Y_CONTROL, Y_ANTI_CONTROL — record their rows.
2. **Identify input register markers:** INPUT_A, INPUT_B, INPUT_R — record their span boundaries (each marker occupies one cell; consecutive INPUT_A cells define a contiguous register span).
3. **Identify target gates:** all non-marker, non-NONE gates.
4. **Build controlMask / antiControlMask** from step 1.
5. **Identify span gates:** QFT, QFT_DG, REVERSE, PHASE_GRADIENT — their span is `{headRow … spanContinuationEnd}` from `spanOverride`.
6. **Validate:** no gate occupies the same row as a control marker in the same column. No two non-marker gates on different rows in the same column unless one is a span continuation (this is enforced at edit time, not simulation time).

### 5.3 Circuit History (Undo/Redo)

- **Past stack:** `std::deque<CircuitGrid>` with max depth **50**
- **Future stack:** `std::deque<CircuitGrid>` — cleared on any new edit

Operations:
- `pushState(newGrid)`: deep-copy current grid onto past, clear future, set current to newGrid. If newGrid equals current (by full comparison), do nothing.
- `undo()`: move current to future (front), pop past top to current.
- `redo()`: move current to past (back), pop future front to current.
- **Drag transactions:** on drag start, snapshot current grid. On drag end (mouseRelease), if grid changed from snapshot, push snapshot to past (making the entire drag a single undo step).

**Grid equality check:** compare row count, column count, then for each cell: `gate` enum value, `params.angle`, `params.customMatrix`, `params.spanOverride`, `params.isSpanContinuation`. JSON-serialization comparison is acceptable but slower; prefer structural comparison.

### 5.4 Circuit Serialization (File Format)

File extension: `.qmlsuite.json`

JSON schema:
```json
{
  "version": "1.0",
  "metadata": {
    "name": "string (non-empty)",
    "description": "string (optional)",
    "createdAt": "ISO 8601 timestamp"
  },
  "circuit": {
    "rows": "integer ≥ 1",
    "cols": "integer ≥ 1",
    "grid": "array[rows] of array[cols] of Cell"
  },
  "customGates": [
    {
      "label": "string",
      "matrix": "[[{re,im},{re,im}],[{re,im},{re,im}]]"
    }
  ]
}
```

Cell JSON:
```json
{
  "gate": "GateType enum string or null",
  "id": "uuid string",
  "params": {
    "angle": "number (optional)",
    "angleExpression": "string (optional)",
    "customMatrix": "[[{re,im},{re,im}],[{re,im},{re,im}]] (optional)",
    "customLabel": "string (optional)",
    "spanOverride": {"startRow": "int", "endRow": "int"} "(optional)",
    "isSpanContinuation": "bool (optional)"
  }
}
```

**Validation on load:** check version is "1.0", check all gate strings against the GateType enum, check matrix entries are numbers, check spanOverride bounds are within [0, rows-1] and startRow ≤ endRow. Reject (show error dialog) on any validation failure. Warn (non-fatal) if a gate type is unknown (treat as NONE).

**Backward compatibility:** the `.qbw.json` format from qbit-weaver must be importable. Its schema is identical except: max 8 rows, no QML-specific fields. Import by reading the grid and ignoring unknown gate types.

---

## 6. QML Module (`qml/`)

### 6.1 Overview

The QML module provides the machinery for training quantum circuits as machine learning models. The core primitive is **differentiable circuit execution**: given a PQC with parameters θ, compute the expectation value ⟨O⟩(θ) and its gradient ∇_θ⟨O⟩ efficiently.

All QML computations use the same simulation engine from `core/engine/`. The QML module adds:
- Parameter management
- Gradient computation (parameter shift rule, adjoint differentiation)
- Optimizers
- Data encoding schemes
- High-level algorithm implementations

### 6.2 Parameterized Quantum Circuit (PQC)

```cpp
class PQC {
public:
    // Circuit ansatz: a CircuitGrid where some rotation gates have a special
    // parameter index instead of a fixed angle
    struct ParameterRef {
        int row, col;        // cell location in grid
        int paramIndex;      // index into theta vector
    };

    CircuitGrid baseGrid;                     // Gate structure (fixed)
    std::vector<ParameterRef> paramRefs;      // Which cells are trainable
    std::vector<double> theta;                // Current parameter values

    // Build a concrete CircuitGrid by substituting current theta values
    CircuitGrid materialize() const;

    // Simulate and return final state
    StateVec run(int numQubits) const;

    // Compute expectation value of observable O given current theta
    double expectation(const Observable& O, int numQubits) const;

    int numParams() const { return static_cast<int>(theta.size()); }
};
```

**Observable:** a weighted sum of Pauli tensor products.
```cpp
struct PauliTerm {
    std::vector<std::pair<int, char>> ops;  // {qubit_index, 'X'|'Y'|'Z'|'I'}
    double coefficient;
};
using Observable = std::vector<PauliTerm>;
```

Expectation value computation for a Pauli string P on n qubits:
- For each Pauli operator in P, apply the corresponding basis rotation (e.g. H for X, H·S† for Y) to the state, then compute the Z-basis expectation of the affected qubits.
- The full Pauli string expectation is `Re(⟨ψ|P|ψ⟩)` computed via the efficient inner product:
  ```
  ⟨P⟩ = Σ_i sign(i) × |α_i|²
  ```
  where `sign(i)` is `+1` if the parity of the bits at Pauli-Z positions is even, `-1` if odd, after basis rotations.

### 6.3 Parameter Shift Rule

For a gate `G(θ) = exp(-iθ/2 P)` where P is a Pauli operator (X, Y, or Z):

```
∂⟨O⟩/∂θ_k = [⟨O⟩(θ + π/2 · e_k) - ⟨O⟩(θ - π/2 · e_k)] / 2
```

Implementation:
```cpp
std::vector<double> parameterShiftGradient(
    const PQC& circuit,
    const Observable& O,
    int numQubits
) {
    std::vector<double> grad(circuit.numParams());
    for (int k = 0; k < circuit.numParams(); k++) {
        auto theta_plus  = circuit.theta; theta_plus[k]  += M_PI / 2;
        auto theta_minus = circuit.theta; theta_minus[k] -= M_PI / 2;
        PQC c_plus  = circuit; c_plus.theta  = theta_plus;
        PQC c_minus = circuit; c_minus.theta = theta_minus;
        grad[k] = (c_plus.expectation(O, numQubits) - c_minus.expectation(O, numQubits)) / 2.0;
    }
    return grad;
}
```

This requires `2 × numParams` circuit evaluations. For circuits with many parameters, offer adjoint differentiation as an alternative (see §6.4).

The parameter shift rule is exact for gates of the form `exp(-iθ/2 P)`. For other parameterized gates, fall back to finite differences with step `h = 1e-5`:
```
grad[k] ≈ (⟨O⟩(θ + h·e_k) - ⟨O⟩(θ - h·e_k)) / (2h)
```

### 6.4 Adjoint Differentiation (Advanced)

Adjoint differentiation computes the full gradient vector in **1 forward pass + 1 backward pass** (plus checkpoint recomputation overhead), compared to the parameter shift rule's `2L` forward passes for `L` parameters. This is the primary advantage for circuits with many parameters.

**Complexity:**
- Time: `O(G)` for the forward pass and `O(G)` for the backward pass, where `G` is the total number of gates. Each parameter's gradient is extracted during the backward sweep at O(2^n) cost, giving `O(G × 2^n + L × 2^n)` total — effectively `O(G × 2^n)` since `L ≤ G`.
- Memory: `O(G × 2^n)` if all intermediate states are stored naively; reduced to `O(C × 2^n + (G/C) × 2^n)` with checkpointing every `C` gates (recompute from nearest checkpoint during backward pass).

**Algorithm:**

1. **Forward pass:** apply gates `G_1, G_2, ..., G_K` sequentially to `|0⟩`, recording intermediate states `|ψ_0⟩, |ψ_1⟩, ..., |ψ_K⟩` (or checkpoints every 10 gates — see below).
2. **Initialize adjoint state:** `|λ⟩ = O|ψ_K⟩` where `O` is the observable operator.
3. **Backward pass:** for `k = K` down to `1`:
   - If gate `k` has a trainable parameter `θ_j`:
     ```
     ∂⟨O⟩/∂θ_j = 2 × Re(⟨λ| (∂G_k/∂θ_j) |ψ_{k-1}⟩)
     ```
   - Propagate the adjoint state backward: `|λ⟩ ← G_k† |λ⟩`
   - Recover `|ψ_{k-1}⟩` from the nearest checkpoint by replaying gates forward from that checkpoint.

**Gate derivative matrices** (must be implemented for each trainable gate type):

| Gate | `G(θ)` | `∂G/∂θ` |
|------|--------|----------|
| `Rx(θ)` | `exp(-iθX/2)` | `(-i/2) X · Rx(θ)` = `[[-sin(θ/2)/2, -i·cos(θ/2)/2], [-i·cos(θ/2)/2, -sin(θ/2)/2]]` |
| `Ry(θ)` | `exp(-iθY/2)` | `(-i/2) Y · Ry(θ)` = `[[-sin(θ/2)/2, -cos(θ/2)/2], [cos(θ/2)/2, -sin(θ/2)/2]]` |
| `Rz(θ)` | `exp(-iθZ/2)` | `(-i/2) Z · Rz(θ)` = `[[-i·exp(-iθ/2)/2, 0], [0, i·exp(iθ/2)/2]]` |

For any other parameterized gate that is not of the form `exp(-iθP/2)` for a Pauli `P`, the derivative must be computed via finite differences or by decomposing into native rotation gates first.

**Checkpointing:** store intermediate states every 10 gates. During the backward pass, recover `|ψ_{k-1}⟩` by replaying from the nearest earlier checkpoint. This limits peak memory to `O((G/10 + 10) × 2^n)` — at 16 qubits and 200 gates, roughly `(20 + 10) × 1 MB ≈ 30 MB`.

### 6.5 Optimizers

All optimizers implement:
```cpp
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<double>& params,
                      const std::vector<double>& grad) = 0;
    virtual void reset() = 0;
};
```

**Adam:**
```
m ← β1 × m + (1-β1) × grad
v ← β2 × v + (1-β2) × grad²
m_hat = m / (1 - β1^t)
v_hat = v / (1 - β2^t)
params -= lr × m_hat / (√v_hat + ε)
```
Default: `lr=0.01`, `β1=0.9`, `β2=0.999`, `ε=1e-8`.

**SGD with momentum:**
```
velocity ← momentum × velocity - lr × grad
params += velocity
```
Default: `lr=0.01`, `momentum=0.0`.

**SPSA (Simultaneous Perturbation Stochastic Approximation):**
For gradient-free optimization. Uses random perturbation vectors:
```
Δ ~ Rademacher(±1)^n
grad_approx = (f(θ + c·Δ) - f(θ - c·Δ)) / (2c) × Δ^{-1}
```
where `c` decays as `c / (k + 1)^γ` with `c=0.1`, `γ=0.101`.
Default: `lr=0.05` decaying as `lr/(k+1)^α`, `α=0.602`.

**COBYLA (Constrained Optimization BY Linear Approximation):**
Gradient-free, wraps the nlopt or similar library. Suitable for noisy expectation values from shot-based simulation.

### 6.6 Data Encoding Schemes

```cpp
class Encoding {
public:
    virtual ~Encoding() = default;
    // Returns a CircuitGrid that encodes classical data x into quantum state
    // The encoding circuit is prepended to the main ansatz
    virtual CircuitGrid encode(const std::vector<double>& x, int numQubits) const = 0;
};
```

**Angle Encoding:**
For input vector `x ∈ ℝ^n` (n ≤ numQubits), apply `Ry(x[i])` to qubit `i`. Supports repeated encoding layers.

**Amplitude Encoding:**
Encode `x ∈ ℝ^(2^n)` (normalized) directly as the state vector amplitudes. Use the Mottonen state preparation algorithm (a sequence of multi-controlled rotations). This requires `O(2^n)` gates in the worst case.

**IQP Encoding (Instantaneous Quantum Polynomial):**
Apply `H` to all qubits, then `Rz(x[i])` to qubit `i`, then `Rzz(x[i]×x[j])` for adjacent pairs, then `H` again.
`Rzz(φ)` on qubits (i,j): CNOT(i→j), Rz(φ) on j, CNOT(i→j).

**Basis Encoding:**
For binary input `x ∈ {0,1}^n`, apply X to qubit i if `x[i] = 1`.

### 6.7 Quantum Kernel

The quantum kernel between data points `x_i` and `x_j` is:
```
K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
```
where `|φ(x)⟩ = U(x)|0⟩` is the encoded state.

Computed by: simulate `U(x_j)†U(x_i)|0⟩` and measure the probability of the all-zeros outcome.

```cpp
class QuantumKernel {
public:
    Encoding* encoding;
    int numQubits;
    double compute(const std::vector<double>& xi,
                   const std::vector<double>& xj) const;
    // Returns n×n kernel matrix for dataset
    std::vector<std::vector<double>> kernelMatrix(
        const std::vector<std::vector<double>>& X) const;
};
```

**Numerical stability of the kernel matrix:** floating-point errors can cause the kernel matrix to violate the mathematical properties that downstream consumers (SVM solver, eigendecomposition) depend on:

1. **Diagonal clamping:** `K(x_i, x_i)` should theoretically be exactly `1.0`, but simulation noise can push it slightly above or below. After computing the full matrix, force all diagonal entries to exactly `1.0`.
2. **Symmetry enforcement:** compute only the upper triangle (`K(x_i, x_j)` for `i ≤ j`) and mirror to the lower triangle, rather than computing both independently. This guarantees exact symmetry.
3. **Positive semidefinite regularization:** add a small ridge term `K += εI` for `ε = 1e-8` before passing to the SVM solver. This absorbs small negative eigenvalues caused by floating-point drift and prevents the SMO solver from failing or producing nonsensical support vectors.
4. **Entry clamping:** clamp all entries to `[0.0, 1.0]` since `|⟨φ(x_i)|φ(x_j)⟩|² ∈ [0, 1]` by definition.

### 6.8 Built-In Algorithm Implementations

Each algorithm is a self-contained class that uses the PQC and optimizer infrastructure.

#### VQE (Variational Quantum Eigensolver)

Finds ground state energy of a Hamiltonian H:
```
Minimize ⟨ψ(θ)|H|ψ(θ)⟩ over θ
```
- Input: Observable H (sum of Pauli terms), ansatz type, numQubits, optimizer
- Output: minimum eigenvalue estimate, optimal θ, energy history
- Convergence: stop when `|E(t) - E(t-1)| < tol` for 10 consecutive steps or max 1000 iterations

#### QAOA (Quantum Approximate Optimization Algorithm)

Approximates solutions to combinatorial optimization (e.g. MaxCut):
- Input: problem graph (edge list, weights), depth p, numQubits
- Circuit: alternating problem unitary `U_C(γ) = exp(-iγC)` and mixer `U_B(β) = exp(-iβB)` layers
- Parameters: `[γ_1, β_1, ..., γ_p, β_p]`
- Optimize with COBYLA or Adam

For MaxCut: `C = Σ_{(i,j)∈E} w_{ij}(1 - Z_i Z_j)/2`. Each ZZ term implemented as `CNOT(i,j)·Rz(2γw)·CNOT(i,j)`.

#### QNN (Quantum Neural Network)

Supervised classification using a PQC:
- Input: dataset `(X, y)`, encoding scheme, ansatz, loss function
- Forward pass: encode x → quantum state → measure expectation values of Z_0, Z_1, ... → apply softmax → predicted class probabilities
- Loss: cross-entropy `L = -Σ y_k log(p_k)`
- Backward pass: parameter shift rule on each expectation value
- Training loop: minibatch gradient descent

For binary classification (1 output qubit):
```
prediction = (⟨Z_0⟩ + 1) / 2   // maps [-1,1] to [0,1]
```

#### QSVM (Quantum Support Vector Machine)

Uses quantum kernel with classical SVM solver:
1. Compute quantum kernel matrix K on training data
2. Solve dual SVM: maximize `Σ α_i - (1/2)Σ α_i α_j y_i y_j K(x_i, x_j)` subject to `Σ α_i y_i = 0`, `0 ≤ α_i ≤ C`
3. Predict: `f(x) = sign(Σ α_i y_i K(x_i, x) + b)`

SVM solver: implement SMO (Sequential Minimal Optimization) or use a bundled quadratic programming solver.

#### QPCA (Quantum PCA)

Uses quantum phase estimation (QPE) to extract eigenvalues of a density matrix encoding of classical data. This algorithm is primarily educational in a simulation context — real speedup requires fault-tolerant hardware. However, it demonstrates the QPE subroutine and density matrix exponentiation in a concrete setting.

**Input:**
- Classical data matrix `X` of shape `(m, d)` (m samples, d features)
- `numDataQubits`: number of qubits to encode the covariance matrix (must satisfy `2^numDataQubits ≥ d`; pad with zeros if needed)
- `numPEQubits`: number of qubits for the phase estimation register (controls eigenvalue precision: `numPEQubits` bits of precision)

**Procedure:**
1. Compute the covariance matrix `Σ = (1/m) X^T X` classically.
2. Normalize: `ρ = Σ / Tr(Σ)` so that `ρ` is a valid density matrix (positive semidefinite, unit trace). If `Σ` has negative eigenvalues due to floating-point error, clamp them to zero and renormalize.
3. Construct the unitary `U = exp(2πi ρ)` via eigen-decomposition: decompose `ρ = V Λ V†`, then `U = V exp(2πi Λ) V†`. This is computed classically (using Eigen3) and then decomposed into a gate sequence.
4. **Gate decomposition of U:** decompose the `2^numDataQubits × 2^numDataQubits` unitary into single- and two-qubit gates using the Solovay-Kitaev algorithm or, for small `numDataQubits ≤ 4`, direct matrix decomposition via Cosine-Sine decomposition (CSD). For v1.0, limit `numDataQubits ≤ 4` and use CSD.
5. **QPE circuit:** standard QPE using `numPEQubits` ancilla qubits with controlled-`U^(2^k)` gates and inverse QFT on the ancilla register.
6. Measure the ancilla register. Repeated shots yield eigenvalue estimates `λ_k ≈ (measurement integer) / 2^numPEQubits`.
7. For each distinct measured eigenvalue, the post-measurement state of the data register approximates the corresponding eigenvector.

**Output:** list of `(eigenvalue_estimate, eigenvector_amplitudes)` pairs, sorted by eigenvalue magnitude descending (largest principal components first).

**Limitations for v1.0:** `numDataQubits` capped at 4, meaning covariance matrices up to 16×16. Combined with the PE register, total qubit count can reach 4 + 8 = 12, well within the 16-qubit budget. The CSD decomposition produces `O(4^n)` gates for `n` data qubits, which is tractable at `n = 4` (~256 gates) but not beyond.

Note: for the simulation context, QPCA is primarily educational; real speedup requires hardware.

### 6.9 Training Script Interface (Python via pybind11)

The Python module exposes:
```python
import qml_suite as qs

# Low-level
sim = qs.Simulator(num_qubits=12)
sim.apply_gate(qs.GateType.H, row=0)
sim.apply_gate(qs.GateType.CX, row=0, controls=[0], target=1)
state = sim.get_state()   # numpy complex128 array of length 2^n
bloch = sim.bloch_vector(qubit=0)

# PQC
pqc = qs.PQC(num_qubits=8, ansatz=qs.Ansatz.HardwareEfficient, layers=3)
pqc.theta = np.random.uniform(-np.pi, np.pi, pqc.num_params)
obs = qs.Observable.from_pauli_string("ZZI")
energy = pqc.expectation(obs)
grad = qs.parameter_shift_gradient(pqc, obs)

# Training
optimizer = qs.Adam(lr=0.01)
for epoch in range(500):
    grad = qs.parameter_shift_gradient(pqc, obs)
    optimizer.step(pqc.theta, grad)

# Kernel
kernel = qs.QuantumKernel(encoding=qs.AngleEncoding(), num_qubits=4)
K = kernel.matrix(X_train)  # returns numpy array
```

All numpy arrays use `float64`/`complex128`. State vector returned as `np.ndarray` of shape `(2^n,)` with dtype `complex128`.

### 6.10 Barren Plateau Mitigation

At 16 qubits with deep, highly-entangling ansätze (particularly Strongly Entangling Layers, §7.5), the variance of parameter gradients decays exponentially with qubit count — the "barren plateau" phenomenon. Without mitigation, VQE, QNN, and QAOA training will silently stall (gradients indistinguishable from zero at `float64` precision) and the user will see no convergence.

**Required mitigations (implement all three):**

1. **Identity-block initialization:** initialize all trainable parameters such that the circuit approximates the identity transformation at the start of training. For `Ry(θ)` and `Rz(θ)` gates, this means initializing `θ ≈ 0` with small random perturbation `θ ~ U(-ε, ε)` for `ε = 0.01`. This keeps the initial state near `|0⟩` and ensures gradients are large at the start. This should be the default initialization strategy for all ansätze; random `U(-π, π)` initialization should be available but not the default.

2. **Layerwise training:** provide an option to train one ansatz layer at a time, freezing all other layers. The training loop adds a new trainable layer every `M` iterations (configurable, default 50). Once all layers are active, continue training all parameters jointly. This avoids the deep-circuit gradient vanishing at initialization.

3. **Gradient magnitude monitoring and early warning:** during training, compute `||∇L||₂` at each step (already shown in the training dashboard, §8.6). If the gradient norm drops below `1e-8` for 20 consecutive iterations while the loss has not converged, emit a warning to the training log and UI: `"Warning: gradient magnitude near zero — possible barren plateau. Consider reducing circuit depth, switching to layerwise training, or using SPSA optimizer."` This is a diagnostic, not a fix, but prevents silent failure.

**Configuration in `TrainingConfig`:**
```cpp
struct TrainingConfig {
    // ... existing fields ...
    enum class InitStrategy { IdentityBlock, Random };
    InitStrategy initStrategy = InitStrategy::IdentityBlock;
    double initEpsilon = 0.01;         // perturbation range for IdentityBlock
    bool layerwiseTraining = false;
    int layerwiseWarmupIters = 50;     // iterations per layer before adding the next
    double barrenPlateauWarnThreshold = 1e-8;
    int barrenPlateauWarnWindow = 20;  // consecutive steps below threshold
};
```

### 6.11 Reproducibility and RNG Seeding

For QML research, exact reproducibility of training runs is critical. All sources of randomness in the application must be seedable from a single global seed.

**Sources of randomness:**
- Measurement collapse (Born rule sampling, §3.10)
- Shot-based expectation value estimation
- SPSA perturbation vectors (§6.5)
- Parameter initialization (random or identity-block with perturbation)
- Minibatch shuffling in QNN training
- Train/test split in data loading

**Seeding strategy:**

```cpp
class RngManager {
public:
    // Initialize from a global seed. If seed is std::nullopt, use std::random_device.
    static void init(std::optional<uint64_t> globalSeed);

    // Get a thread-local generator seeded deterministically from the global seed.
    // Each thread gets seed = hash(globalSeed, threadIndex) to ensure
    // parallel runs are reproducible regardless of thread scheduling.
    static std::mt19937_64& threadLocal();

    // Get the global seed that was used (for logging / experiment tracking).
    static uint64_t getGlobalSeed();
};
```

**Configuration:** the global seed is set via `simulation.globalRngSeed` in `settings.json`. A value of `null` means non-deterministic (use `std::random_device`). The Python bindings expose `qs.set_seed(42)` and `qs.get_seed()`.

**Per-training-job override:** the training configuration panel (§10.3) includes an optional seed field. If set, it overrides the global seed for that training run only. The seed used is recorded in the training log output so experiments can be reproduced.

**Logging:** at the start of every simulation run and every training job, log the effective seed at `spdlog::info` level: `"RNG seed: 12345"`.

---

## 7. Ansatz Library (`qml/pqc/`)

All ansätze are parameterized by `numQubits` and `numLayers`. They return a `CircuitGrid` with `ParameterRef` entries.

### 7.1 Hardware-Efficient Ansatz (HEA)

Each layer consists of:
1. `Ry(θ)` on every qubit (numQubits parameters)
2. `Rz(φ)` on every qubit (numQubits parameters)
3. CNOT ladder: `CX(0→1), CX(1→2), ..., CX(n-2→n-1)`

Total parameters: `numLayers × 2 × numQubits`.

### 7.2 Real Amplitudes Ansatz

Each layer:
1. `Ry(θ)` on every qubit (real amplitudes only)
2. CNOT ladder

No Rz gates ⟹ state vector remains real-valued throughout. Useful for energy minimization problems where the ground state is real.

### 7.3 QAOA Ansatz

Fixed structure (see §6.8). Parameters are `[γ_1, β_1, ..., γ_p, β_p]`.

### 7.4 Two-Local Ansatz

Each layer:
1. Single-qubit rotation `R(θ)` (RX, RY, or RZ, configurable) on each qubit
2. Two-qubit entangling gates (CX, CZ, or SWAP) on all pairs (linear, circular, or full connectivity)

### 7.5 Strongly Entangling Layers

Each layer applies three single-qubit rotations (Rz-Ry-Rz Euler decomposition) to every qubit, then CNOT(i, (i+d)%n) for stride d (d=1 in layer 1, d=2 in layer 2, etc.). Highly expressive; from PennyLane StronglyEntanglingLayers.

---

## 8. Visualization Suite (`ui/Visualization/`)

### 8.1 Bloch Sphere

Rendered using OpenGL 4.5 core profile (or Qt3D as a fallback). One sphere widget per qubit, displayed in a scrollable row.

**Geometry:**
- Sphere radius normalized to 1.0
- Three axes: Z (vertical, up = |0⟩), X (horizontal-right), Y (into screen, right-hand coordinate system)
- Equatorial circle as a thin wire ring
- North pole label: `|0⟩`, south pole: `|1⟩`, X axis: `|+⟩`, Y axis: `|i⟩`
- State vector as an arrow from origin to Bloch point `(X, Y, Z)`

**Coloring:**
- Arrow color: interpolate HSV hue from 240° (blue, Z=+1 ground state) to 0° (red, Z=−1 excited state) based on Z coordinate
- Sphere surface: translucent gray wireframe

**Interaction:** hover shows tooltip with exact `(X, Y, Z)` values and Dirac notation `α|0⟩ + β|1⟩` where `α = cos(θ/2)`, `β = sin(θ/2)·exp(iφ)`, `θ = arccos(Z)`, `φ = atan2(Y, X)`.

### 8.2 Amplitude Heatmap

Displays all `2^n` basis states and their amplitudes. At 16 qubits (65,536 states) it is not practical to display each state as a cell. Two modes:

**Compact mode (≤ 12 qubits, ≤ 4096 cells):**
- Grid layout: `rowCount = 2^(n/2)` rows, `colCount = 2^(n - n/2)` columns (ceiling division for rows)
- Each cell background: deep purple (`#1A1030`)
- Fill height proportional to `|α_i|²` (probability), color cyan
- Phase indicator: thin white line from cell center at angle `arg(α_i)`, length proportional to `|α_i|`
- Cell label: binary string of basis state, displayed on hover only for readability

**Hierarchical mode (13–16 qubits):**
- Treemap layout grouping qubits in blocks of 4
- Top level: `2^4 = 16` tiles for the 4 MSBs
- Each tile subdivided into `2^4 = 16` sub-tiles for the next 4 bits, etc.
- Tile color represents total probability mass in that block
- Click to drill down into sub-tiles; breadcrumb shows current prefix

**Performance:** update at most 30 fps. If the simulation is running at 60 fps (animated gates), skip amplitude heatmap updates on alternate frames.

### 8.3 Probability Bar Chart

Vertical bar chart showing `P(i) = |α_i|²` for each basis state. At 16 qubits, show only the top-32 most probable states, labeled by their binary representation. Update whenever the state changes.

### 8.4 Entanglement Map

`n×n` heatmap where cell `(i,j)` shows the mutual information between qubits `i` and `j`:
```
I(i:j) = S(ρ_i) + S(ρ_j) - S(ρ_ij)
```
where `S(ρ) = -Tr(ρ log ρ)` is the von Neumann entropy, `ρ_i` is the reduced density matrix of qubit `i` (obtained by tracing out all other qubits), and `ρ_ij` is the reduced density matrix of the qubit pair.

Reduced density matrix of qubit `i`:
```
ρ_i[0][0] = Σ |α_idx|² where bit i of idx = 0    (prob of |0⟩)
ρ_i[1][1] = Σ |α_idx|² where bit i of idx = 1    (prob of |1⟩)
ρ_i[0][1] = Σ conj(α_idx0) × α_idx1              (coherences)
```
where `idx1 = idx0 ^ (1 << bit_i)`.

Von Neumann entropy: compute eigenvalues `λ_1, λ_2` of `ρ_i` (2×2 matrix → analytic formula), then `S = -Σ λ_k log(λ_k)` (with `0 log 0 = 0`).

This is O(n² × 2^n) and should be computed in a background thread.

### 8.5 Density Matrix View

For the current state `|ψ⟩`, display `ρ = |ψ⟩⟨ψ|` as a heatmap of `|ρ_ij|`. This is a `2^n × 2^n` matrix — only practical for ≤ 8 qubits in the full view. For larger systems, show only the 32×32 block of the most probable basis states.

### 8.6 QML Training Dashboard

When a QML training job is running, display:
- **Loss curve:** real-time line plot of loss vs. epoch (or iteration)
- **Parameter histogram:** distribution of `θ` values, updated every 10 epochs
- **Gradient magnitude plot:** `||∇L||₂` vs. epoch
- **Expectation values panel:** `⟨O_k⟩` for each output observable vs. epoch

Implemented with Qt Charts. Data is streamed from the optimizer's training loop via a `QueuedConnection` signal from the worker thread.

### 8.7 Circuit Diagram Renderer

Renders the circuit grid as a publication-quality diagram:
- Qubit wires: horizontal gray lines, labeled `q_0`, `q_1`, … on the left
- Gate boxes: rounded rectangles with gate symbol, colored by category
- Control dots: filled black circles at CONTROL rows, open circles at ANTI_CONTROL
- Vertical control line connecting control dots to gate box
- SWAP: two × symbols connected by a vertical line
- Span gates (QFT, REVERSE, PHASE_GRADIENT): single box spanning multiple wires

Export to SVG and PNG at configurable resolution (72–600 DPI).

---

## 9. Circuit Editor UI (`ui/CircuitEditor/`)

### 9.1 Grid Widget

The circuit grid is a `QAbstractItemView` subclass with custom painting. Each cell is `CELL_WIDTH × ROW_HEIGHT` pixels:
- `CELL_WIDTH = 56px`
- `ROW_HEIGHT = 72px`

**Painting:**
- Background: dark (`#0D0D1A`)
- Qubit wire: horizontal gray line at vertical center of each row
- Gate cells: painted by `GatePainter` based on `GateType`
- Control lines: vertical lines connecting control dots to gate cells in the same column
- Selected cell: cyan border `2px`
- Hover cell: lighter background

**Interaction (desktop):**
- Drag gate from GateLibrary panel → drop on cell: place gate
- Drag gate from one cell to another: move gate
- Right-click on cell: context menu (Delete, Edit Params, Gate Info)
- Click: select cell
- Delete key on selected cell: remove gate
- Arrow keys: move selection

**Interaction (tablet/touch):**
- Tap gate in library → tap cell to place
- Long-press (500ms) on cell: delete gate
- Pinch to zoom (scale both CELL_WIDTH and ROW_HEIGHT, min 0.5×, max 2×)

### 9.2 Gate Library Panel

Tabbed panel (left or bottom, configurable) with four tabs:

**Standard:** X, Y, Z, H, S, SDG, T, TDG, I, CONTROL, ANTI_CONTROL, X_CONTROL, X_ANTI_CONTROL, Y_CONTROL, Y_ANTI_CONTROL, CX, CZ, SWAP, CCX, SQRT_X, SQRT_X_DG, SQRT_Y, SQRT_Y_DG, MEASURE, REVERSE, BLOCH_VIS, PERCENT_VIS

**Parameterized:** RX, RY, RZ (with angle input), preset RX_PI_2/4/8/12, RY_PI_2/4/8/12, RZ_PI_2/4/8/12, ZT, XT, YT, ZA, XA, YA, ZB, XB, YB, EXP_Z, EXP_X, EXP_Y, QFT, QFT_DG, PHASE_GRADIENT

**Arithmetic:** INC, DEC, ADD_A, SUB_A, MUL_A, DIV_A, MUL_B, DIV_B, INC_MOD_R, DEC_MOD_R, ADD_A_MOD_R, SUB_A_MOD_R, MUL_A_MOD_R, DIV_A_MOD_R, A_LT_B, A_LEQ_B, A_GT_B, A_GEQ_B, A_EQ_B, A_NEQ_B, SCALE_I, SCALE_NEG_I, SCALE_SQRT_I, SCALE_SQRT_NEG_I, INPUT_A, INPUT_B, INPUT_R

**Custom:** list of user-defined 2×2 unitary gates with "New Custom Gate" button

**Search:** text field at top, searches by gate name and description. Results displayed across all tabs. Debounce: 150ms.

### 9.3 Parameterized Gate Angle Dialog

When a parameterized gate (RX, RY, RZ) is placed, a dialog appears:
- Text field: angle expression (e.g. `pi/4`, `3*pi/8`, `0.785`)
- Live preview: shows the numeric value in radians
- Error message shown inline if expression cannot be parsed
- OK button disabled while expression is invalid
- Cancel: place gate with default angle 0

### 9.4 Custom Gate Dialog

Allows defining a 2×2 unitary matrix:
- 2×2 grid of complex number inputs (re + im fields each)
- Label field (max 4 characters for gate display)
- Unitarity check: compute `M†M` and verify it's close to identity (tolerance 1e-6)
- Error if matrix is not unitary: highlight cells in red, show `||M†M - I||_F` value
- On confirm: add to custom gate library and available gates in current circuit

### 9.5 Step Mode / Simulation Timeline

A toolbar strip with: `|◀` (jump to start), `◀` (step back), `▶/⏸` (play/pause), `▶|` (step forward), `▶▶|` (jump to end).

- **Step index 0:** state is `|0…0⟩` (before any gates)
- **Step index k:** state after the k-th active column (columns with at least one gate)
- **Active columns:** precomputed from `runCircuitWithMeasurements`

Playback speed: configurable 200ms–2000ms per step (default 500ms). During playback, the amplitude heatmap and Bloch spheres update to show the state at the current step.

The timeline is only available after "Run" has been executed. While time-parameterized gates are present, the timeline plays at 60fps with t advancing rather than stepping through columns.

### 9.6 Algorithm Templates

A dropdown or sidebar listing pre-built circuits. Organized identically to qbit-weaver's categories:

**Entanglement:** Bell State, GHZ State, W State, Superdense Coding, Quantum Teleportation

**Fundamental Algorithms:** Deutsch, Deutsch-Jozsa, Bernstein-Vazirani, Grover's, Phase Kickback

**Quantum Algorithms:** QFT demo, Shor's Algorithm (modular exponentiation demo)

**Error Correction:** Bit-Flip Code, Phase-Flip Code, Steane [[7,1,3]] Code

**QML Circuits (new):** Hardware-Efficient Ansatz demo, QAOA MaxCut (6 qubits), VQE H2 molecule demo, QNN XOR classifier demo

Each template specifies: name, description, category, and a `CircuitGrid` factory function.

---

## 10. QML Workspace UI (`ui/QMLWorkspace/`)

A secondary panel (separate tab or dockable panel) for QML-specific workflows.

### 10.1 Ansatz Builder

- Dropdown: select ansatz type (HEA, RealAmplitudes, TwoLocal, StronglyEntangling)
- Spinbox: numLayers (1–20)
- Spinbox: numQubits (2–16)
- Entanglement topology: Linear, Circular, Full
- "Generate" button: populates the circuit editor with the selected ansatz
- Parameter count display: "N trainable parameters"

### 10.2 Observable Editor

GUI for building Pauli observables:
- Add terms: select qubit, Pauli type (I/X/Y/Z), coefficient
- Multi-qubit terms: specify a Pauli type per qubit
- Preview: shows the observable as a sum of Pauli strings
- Common presets: Z (single qubit), ZZ (two-qubit), H2 Hamiltonian, MaxCut objective

### 10.3 Training Configuration Panel

- Optimizer: dropdown (Adam, SGD, SPSA, COBYLA)
- Optimizer hyperparameters (fields shown based on selection)
- Gradient method: Parameter Shift, Adjoint Differentiation, or Finite Difference
- Max iterations / epochs
- Convergence tolerance
- Batch size (for QNN)
- Shots mode: analytic (exact expectation values) or shot-based (N shots per evaluation)
- Initialization strategy: Identity Block (default) or Random (see §6.10)
- Layerwise training: checkbox, with warmup iterations spinbox (default 50, see §6.10)
- RNG seed: optional integer field. If blank, uses the global seed from settings. If set, overrides for this run only (see §6.11)
- "Start Training" button: launches training in a `QThread`, feeding progress to the dashboard

### 10.4 Data Loader

For QNN and QSVM workflows:
- Load CSV: first N columns as features, last column as label
- Built-in datasets: XOR (4 points), Circles (synthetic), Moons (synthetic)
- Feature scaling: MinMax or StandardScaler
- Train/test split ratio
- Preview: scatter plot of first 2 features colored by class

---

## 11. Build System

### 11.1 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Qt6 | ≥ 6.5 | GUI (Widgets, Charts, 3D/OpenGL, Core, Concurrent) |
| Eigen3 | ≥ 3.4 | Matrix algebra for density matrix / observable computation |
| OpenMP | any | Gate application parallelism |
| pybind11 | ≥ 2.11 | Python bindings |
| nlohmann/json | ≥ 3.11 | Circuit file serialization |
| Catch2 | ≥ 3.0 | Unit testing |
| spdlog | ≥ 1.12 | Logging |
| {fmt} | ≥ 10.0 | String formatting (used by spdlog) |

Optional:
| Library | Purpose |
|---------|---------|
| nlopt | COBYLA optimizer backend |
| Intel MKL or OpenBLAS | BLAS backend for Eigen (improves density matrix ops) |

### 11.2 CMake Structure

```cmake
# Top-level
cmake_minimum_required(VERSION 3.25)
project(qml_suite VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

# Core library (no Qt dependency)
add_library(qml_suite_core STATIC ...)
target_link_libraries(qml_suite_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX spdlog::spdlog)

# Main executable
add_executable(qml_suite ...)
target_link_libraries(qml_suite PRIVATE qml_suite_core Qt6::Widgets Qt6::Charts Qt6::OpenGL)

# Python module
pybind11_add_module(qml_suite_python python/bindings/module.cpp)
target_link_libraries(qml_suite_python PRIVATE qml_suite_core)

# Tests
add_executable(tests ...)
target_link_libraries(tests PRIVATE qml_suite_core Catch2::Catch2WithMain)
```

### 11.3 Compiler Flags

```cmake
# Release builds
-O3 -march=native -ffast-math

# Debug / test builds (NO -ffast-math — see note below)
-O0 -g -march=native

# SIMD explicitly for gate loops
-mavx2 -mfma      # for x86_64
-mcpu=native      # for ARM (Apple Silicon)

# OpenMP
-fopenmp
```

**`-ffast-math` policy:** `-ffast-math` enables floating-point reassociation, which means parallel reductions (OpenMP) may sum terms in non-deterministic order, producing results that are *not* bit-identical across thread counts. This directly conflicts with the bit-identical parallelism requirement in §12.1.

Resolution:
- **Release builds:** enable `-ffast-math` for throughput. Relax the parallelism correctness test to a tolerance of `1e-10` rather than bit-identical.
- **Debug / test builds:** disable `-ffast-math`. Use ordered reductions (`#pragma omp parallel for reduction(+:...)` with deterministic scheduling) so that the bit-identical test passes. This ensures correctness is validated without floating-point reordering artifacts.
- In both modes, the simulation does not require IEEE-754 exceptional values (NaN/Inf), so `-ffinite-math-only` and `-fno-signed-zeros` are safe independently if finer-grained control is preferred over the blanket `-ffast-math`.

### 11.4 Platform Targets

- **macOS 14+** (primary development): Apple Silicon and Intel. Use `Accelerate.framework` as BLAS backend for Eigen on macOS.
- **Linux**: Ubuntu 24.04+, Fedora 40+. OpenBLAS or Intel MKL.
- **Windows**: Windows 11, MSVC 2022 or clang-cl. Ensure AVX2 flags work.

---

## 12. Testing Strategy (`tests/`)

### 12.1 Engine Tests

**Gate correctness:** for each fixed gate, verify that applying it to every basis state gives the correct result. Compare against reference values computed from the matrix definition. Tolerance: `1e-12`.

**Normalization:** after applying any sequence of unitary gates, verify `Σ |α_i|² = 1` to tolerance `1e-12`.

**Unitarity of QFT:** apply QFT then QFT† to an arbitrary state; verify it equals the original state to tolerance `1e-10`.

**Measurement:** apply H to qubit 0, measure 10,000 times; verify outcome frequencies are within 3σ of 50/50 using chi-squared test.

**Arithmetic gates:** verify `INC` correctly increments all basis states (including wrap-around). Verify `MUL_A(3)` on a 4-qubit register.

**Bloch vector:** for `|0⟩`: expect `(0,0,1)`. For `|1⟩`: expect `(0,0,-1)`. For `|+⟩ = H|0⟩`: expect `(1,0,0)`. For `|i⟩ = S|+⟩`: expect `(0,1,0)`.

**Dense/sparse equivalence:** for 12-qubit circuits with <10% nonzero amplitudes, verify dense and sparse implementations produce identical states (tolerance `1e-12`).

**Parallelism:** run gate application with 1 thread and N threads.
- In debug builds (no `-ffast-math`): verify results are bit-identical using ordered reductions and deterministic thread partitioning.
- In release builds (`-ffast-math` enabled): verify results agree to tolerance `1e-10`, since floating-point reassociation may alter summation order.
See §11.3 for the full `-ffast-math` policy.

### 12.2 QML Tests

**Parameter shift:** for `Ry(θ)` gate with observable `Z`, verify gradient analytically: `∂⟨Z⟩/∂θ = -sin(θ)`. Compare parameter shift result to analytic at several `θ` values; tolerance `1e-6`.

**Gradient descent convergence:** VQE on a 2-qubit Hamiltonian `H = Z⊗Z` (ground energy -1). Run Adam for 200 iterations starting from random θ; verify final energy < `-0.99`.

**Encoding round-trip:** angle encoding of vector `x`, then decode by computing `⟨Z_i⟩`; verify approximate recovery of `x`.

**Kernel positivity:** quantum kernel matrix must be positive semidefinite (all eigenvalues ≥ -ε for ε=1e-8).

### 12.3 Property-Based Tests

Use a property testing library (libcheck or rapidcheck):
- All gates preserve normalization (apply arbitrary gate to arbitrary state)
- QFT is the inverse of QFT† (arbitrary span, arbitrary state)
- Arithmetic permutations are bijections (for each input, unique output)
- History undo/redo is a proper stack (random sequence of edits, undos, redos)

### 12.4 Regression Tests

Port qbit-weaver's `quantum.test.ts` test cases to Catch2. For each algorithm template circuit, simulate in both apps (qbit-weaver in CI via Node.js, new engine via the test binary) and compare final state vectors to tolerance `1e-10`. This ensures gate-by-gate fidelity with the existing simulator.

---

## 13. Module Boundaries and Layering Rules

```
UI layer          (ui/)           → depends on: qml/, core/
QML layer         (qml/)          → depends on: core/
Core layer        (core/)         → depends on: nothing (only STL, Eigen, OpenMP)
Python bindings   (python/)       → depends on: core/, qml/ (NOT ui/)
```

**Strict rules:**
1. `core/` must compile without Qt. It must compile without Python headers. Verified by building `qml_suite_core` as a standalone static library with only STL/Eigen/OpenMP.
2. `qml/` must not include any Qt headers. All communication with the UI is via signal/slot connections to types defined in `ui/`.
3. The simulation kernel (`core/engine/Simulator.hpp`) must be callable from multiple threads simultaneously (stateless: each call takes a `StateVec` and returns a new `StateVec`; no global mutable state in the hot path).
4. The gate matrix cache (`GateMatrix::getCached(GateType)`) is the only global mutable state in `core/`. It must be protected by `std::call_once` or initialized at program startup before any threads are spawned.
5. **Training thread isolation:** when a QML training job runs in a `QThread` (§10.3), the training thread owns a private copy of the `PQC` (including `theta`). The UI thread must never read `theta` or any mutable training state directly. Instead, the training thread emits snapshots via `QueuedConnection` signals at configurable intervals (default: every iteration). The snapshot is a value type:
   ```cpp
   struct TrainingSnapshot {
       int iteration;
       double loss;
       double gradientNorm;
       std::vector<double> theta;          // copy, not reference
       std::vector<double> expectationValues;
   };
   ```
   The UI dashboard (§8.6) reads only from the most recent snapshot. No mutex is needed because `QueuedConnection` marshals the data across threads via Qt's event loop.

### 13.1 Coding Conventions

**Naming:**
- Types and classes: `PascalCase` (`StateVec`, `PauliTerm`, `CircuitGrid`)
- Functions and methods: `camelCase` (`applyGate`, `materialize`, `expectation`)
- Constants and enum values: `UPPER_SNAKE_CASE` (`GateType::SQRT_X_DG`, `M_PI`)
- Local variables and parameters: `camelCase` (`numQubits`, `controlMask`, `theta`)
- Member variables: `camelCase` with no prefix (`baseGrid`, not `m_baseGrid` or `_baseGrid`)
- File names: `PascalCase.hpp` / `PascalCase.cpp` matching the primary class

**`const` correctness:**
- All function parameters that are not mutated must be `const&` (for non-trivial types) or passed by value (for scalars and small types).
- All methods that do not mutate `this` must be marked `const`.
- Prefer `const` local variables where the value is not reassigned.
- Pointer and span parameters that observe without mutating must be `const`.

**Ownership:**
- Use `std::unique_ptr` for exclusive ownership (e.g., optimizer instances held by a training job).
- Use raw pointers or `std::span` for non-owning references within a single scope.
- Use `std::shared_ptr` only when shared ownership is genuinely required (expected to be rare).
- `Encoding*` in `QuantumKernel` (§6.7) is non-owning; the caller retains ownership. Document this at the declaration site.

**Includes:** use `#pragma once` for header guards. Group includes in order: corresponding header, C++ standard library, third-party libraries, project headers, separated by blank lines.

---

## 14. Error Handling and Fault Strategy

### 14.1 Error Reporting Convention

The codebase uses **exceptions for unrecoverable errors** and **`std::expected<T, Error>` (C++23) for recoverable failures** in the `core/` and `qml/` layers. The UI layer catches both and presents them to the user.

```cpp
// Error type hierarchy
enum class ErrorCategory { Simulation, Circuit, QML, FileIO, Validation };

struct Error {
    ErrorCategory category;
    std::string message;       // Human-readable, suitable for UI display
    std::string detail;        // Technical detail for logging
    std::source_location loc;  // Where the error originated
};

// Use std::expected for operations that can fail gracefully
using SimResult = std::expected<StateVec, Error>;
using FileResult = std::expected<CircuitGrid, Error>;
```

If the compiler does not support `std::expected` (pre-C++23 toolchains), fall back to a `Result<T, Error>` type alias wrapping `std::variant<T, Error>` with `.value()` and `.error()` accessors.

### 14.2 Simulation Errors

**NaN/Inf propagation:** after every column application, check that the state vector norm `Σ|α_i|² ∈ (1 - ε, 1 + ε)` for `ε = 1e-6`. If not, the simulation has become numerically unstable. Log the column index and gate configuration at `spdlog::error` level, halt the simulation, and display a UI error: `"Simulation halted: state vector normalization failed at column N. This usually indicates a non-unitary custom gate or a numerical instability."` Do NOT silently renormalize — the user needs to know their circuit is broken.

**Non-unitary custom gate:** when a user defines a custom gate (§9.4), the unitarity check rejects it at entry time. But if a circuit file is loaded with a custom gate that passes the tolerance `1e-6` but drifts during repeated application, the normalization check above catches it.

**Out-of-range qubit index:** if a gate references a qubit row outside `[0, numQubits)`, throw `std::out_of_range` with a message identifying the gate and column. This is a programmer error (circuit validation in §5.2 should prevent it), so an exception is appropriate.

### 14.3 QML Training Errors

**Optimizer divergence:** if the loss value exceeds `1e10` or becomes NaN at any training step, halt training and report: `"Training halted: loss diverged at iteration N. Consider reducing the learning rate or checking the observable definition."` Log the parameter values and gradient at the point of divergence for debugging.

**Gradient NaN:** if any element of the gradient vector is NaN or Inf, halt training with a diagnostic. Common cause: a parameter value that produces a singular or near-singular state.

**Non-convergence timeout:** if training reaches `maxIterations` without the convergence criterion being met, report a warning (not an error) and return the best parameters found so far (lowest loss observed).

### 14.4 File I/O Errors

File load failures (missing file, malformed JSON, schema violations) return `Error` via `std::expected`. The UI shows a dialog with the error message. Partial failures (e.g., unknown gate type treated as NONE) produce warnings logged at `spdlog::warn` and shown as a non-blocking toast notification in the UI.

File save failures (permission denied, disk full) surface immediately as a modal dialog with the OS error message.

### 14.5 Logging

All error and warning paths log through `spdlog` at the appropriate level (`error`, `warn`, `info`). Log output goes to `~/.config/qml_suite/qml_suite.log` with daily rotation and a 10 MB size cap. Debug builds additionally log to `stderr`.

---

## 15. Configuration and Settings

Stored in `~/.config/qml_suite/settings.json` (cross-platform via `QStandardPaths`):

```json
{
  "ui": {
    "theme": "dark",
    "gatePanelPosition": "bottom",
    "cellWidth": 56,
    "rowHeight": 72,
    "defaultNumQubits": 8,
    "defaultNumCols": 10
  },
  "simulation": {
    "autoRunOnEdit": true,
    "autoRunDebounceMs": 100,
    "maxQubits": 16,
    "parallelThreshold": 10,
    "sparseThreshold": 0.10,
    "sparseMinQubits": 12,
    "sparseConversionCooldown": 3,
    "forceStateFormat": "auto",
    "globalRngSeed": null
  },
  "animation": {
    "timeStep": 0.005,
    "fpsTarget": 60
  },
  "recentFiles": ["path/to/circuit.qmlsuite.json"],
  "recentGates": ["H", "CX", "RY"]
}
```

---

## 16. Out of Scope for v1.0

The following are explicitly deferred:

- **Noise simulation** (depolarizing, amplitude damping, shot noise beyond sampling) — future v2
- **Hardware backend** (IBMQ, AWS Braket, IonQ API execution) — future v2
- **Tensor network simulation** (MPS/DMRG) — future v3 (beyond 16 qubits)
- **Collaborative editing** (multi-user circuits)
- **3D circuit layout** (Bloch sphere array for multi-qubit gates in 3D)
- **Symbolic computation** (exact rational/algebraic arithmetic)
- **GPU acceleration** (CUDA/Metal backend) — future v1.5

---

## 17. Open Questions (To Resolve Before Implementation Starts)

1. **UI framework final decision:** Qt6 Widgets vs. Qt6 Quick (QML). Widgets give precise control over the custom circuit grid painter; Quick gives smoother animations. Recommendation: Widgets for the circuit grid (proven pattern for this kind of editor), Quick for the training dashboard animations.

2. **Sparse state format:** ~~`std::unordered_map<int, std::complex<double>>` vs. sorted `std::vector<std::pair<int, std::complex<double>>>`.~~ **RESOLVED: use sorted vector.** Gate application iterates all nonzero entries on every call, making cache-coherent sequential iteration the dominant access pattern. The sorted vector's `O(log n)` lookup cost is irrelevant since point lookups are rare (only needed for control mask checks, which can be batched). The hash map's cache-hostile iteration pattern is a measurable penalty at the densities (1–10%) where sparse mode activates. Update the `SparseState` definition in §3.1 accordingly:
   ```cpp
   struct SparseState {
       int numQubits;
       std::vector<std::pair<int, std::complex<double>>> entries; // sorted by basis-state index
   };
   ```
   Maintain sort invariant via `std::lower_bound` on insertion. After a full gate pass that produces a new vector, sort once at the end rather than maintaining order incrementally.

3. **QFT threshold for FFT vs. matrix multiply:** build the dense `2^s × 2^s` matrix and multiply for spans up to s=8 (matrix size 256×256 = 65536 multiplies); use Cooley-Tukey for s>8.

4. **pybind11 GIL handling:** simulation functions must release the GIL during long computations to allow Python threads to run. Mark all `Simulator::run()` calls with `py::gil_scoped_release`.

5. **History storage at 16 qubits:** storing every column's 1 MB state vector in a 50-deep history stack = 50 MB peak. Acceptable? Or implement checkpoint-based recomputation (store every 5th state, recompute from nearest checkpoint on undo)?
