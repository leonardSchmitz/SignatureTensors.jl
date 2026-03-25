# SignatureTensors.jl

**SignatureTensors.jl** is a Julia package for computing and manipulating signature tensors of paths and membranes, built on top of the [OSCAR](https://www.oscar-system.org) computer algebra system.

> **Version:** 0.1 — No support is guaranteed.

---

## Overview

Path signatures are fundamental objects in rough path theory that capture the essential geometry of sequential data. This package provides a general algebraic framework for computing truncated signatures and working with them symbolically, leveraging the full power of OSCAR's ring arithmetic, Gröbner bases, and Lie theory.

The package supports:

- **Path signatures** — iterated-integrals signatures of piecewise linear, polynomial, spline, axis, and monomial paths over arbitrary OSCAR rings.
- **Membrane signatures** — two-parameter (id-)signatures of piecewise bilinear and polynomial membranes, extending the one-parameter theory.
- **Tensor learning** — recovery of paths from their signature tensors via polynomial systems and Gröbner bases, including the efficient algorithm from [Sch25].
- **Lie group barycenters** — computation of Fréchet means on the free nilpotent Lie group $G_{d,k}$, with several algorithm options including BCH-based and polynomial map approaches.
- **Algebraic operations** — group multiplication, inverse, logarithm, exponential, and graded projections on truncated tensor algebra elements.

All constructions work over arbitrary OSCAR rings, making them compatible with symbolic computation over $\mathbb{Q}$, polynomial rings, rational function fields, and more.

---

## Installation

```julia
] activate .
] instantiate
```

Then load the package alongside OSCAR:

```julia
using Oscar
using SignatureTensors
```

---

## Quick Start

```julia
using Oscar, SignatureTensors

# Define a truncated tensor algebra: dimension d=2, truncation level k=3
d, k = 2, 3
T = TruncatedTensorAlgebra(QQ, d, k)

# Signature of the canonical axis path
sig(T, :axis)

# Signature of a polynomial path t ↦ (t + 2t², 3t + 4t²)
sig(T, :poly, coef = QQ.([1 2; 3 4]))

# Path recovery from a signature tensor
d, k, m = 2, 4, 4
A = QQ.([6 -2 6 -10; 7 -4 10 -4])
S = sig(TruncatedTensorAlgebra(QQ, d, k), :pwln, coef = A)
recover(S)
```

---

## Documentation

```@contents
Pages = ["api.md"]
Depth = 2
```

---

## References

This package implements and extends methods from the following works:

- Améndola, Friz, Sturmfels — *Varieties of Signature Tensors*, Forum of Mathematics Sigma, 2019.
- Pfeffer, Seigal, Sturmfels — *Learning Paths from Signature Tensors*, SIAM J. Matrix Anal., 2019.
- Clausel et al. — *The Barycenter in Free Nilpotent Lie Groups*, SIAM J. Appl. Algebra Geom., 2024.
- Améndola, Schmitz — *Learning Barycenters from Signature Matrices*, arXiv:2509.07815, 2025.
- Lotter, Schmitz — *Signature Matrices of Membranes*, Algebraic Statistics, 2026.
- Schmitz — *An Efficient Algorithm for Tensor Learning*, arXiv:2512.14218, 2025.

The full reference list is available in the accompanying paper:
> G. Riffo and L. Schmitz, *Signature Tensors in OSCAR*, 2025.  
> Code: [github.com/leonardSchmitz/signature-tensors-in-OSCAR](https://github.com/leonardSchmitz/signature-tensors-in-OSCAR)

---

## Authors

| Name | Affiliation | Email |
|------|-------------|-------|
| Gabriel Riffo | TU Berlin | [riffo@tu-berlin.de](mailto:riffo@tu-berlin.de) |
| Leonard Schmitz | TU Berlin | [lschmitz@math.tu-berlin.de](mailto:lschmitz@math.tu-berlin.de) |

Funded by the Deutsche Forschungsgemeinschaft (DFG) — CRC/TRR 388 *"Rough Analysis, Stochastic Dynamics and Related Fields"* — Project A04 and B01, 516748464.
