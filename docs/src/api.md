# API Reference

```@docs
SignatureTensors.SignatureTensors
```

---

## Types

The core algebraic structures of the package. `TruncatedTensorAlgebra` defines the ambient
space $T_{d,k}$ together with a coefficient ring and a sequence type (e.g. `:iis` for
iterated-integrals signatures, `:p2id` for two-parameter id-signatures). Elements of this
algebra are represented by `TruncatedTensorAlgebraElem`, which stores the graded components
as a vector of arrays. The free truncated signature algebra types support multivariate
non-commutative constructions used internally.

```@docs
SignatureTensors.TruncatedTensorAlgebra
```

---

## Signature Constructors

The primary interface for computing signatures. `sig` dispatches on the `geom_type` symbol
to select the appropriate path or membrane class, and accepts keyword arguments for
coefficients, shape, composition, regularity, and algorithm choice. See the table below for
supported geometry types.

| `geom_type` | Description | Key arguments |
|-------------|-------------|---------------|
| `:point` | Constant path (unit in $G_{d,k}$) | — |
| `:axis` | Canonical axis path | — |
| `:mono` | Moment path | — |
| `:pwln` | Piecewise linear path | `coef`, `algorithm` (`:Chen` or `:congruence`) |
| `:poly` | Polynomial path | `coef`, `algorithm` (`:congruence` or `:ARS26`) |
| `:pwmon` | Piecewise monomial path | `composition`, `regularity` |
| `:spline` | Piecewise polynomial (spline) path | `coef`, `composition`, `regularity` |
| `:segment` | Single linear segment | `coef` |
| `:axis` *(membrane)* | Canonical axis membrane | `shape` |
| `:mono` *(membrane)* | Monomial membrane | `shape` |
| `:pwbln` | Piecewise bilinear membrane | `coef`, `shape`, `algorithm` (`:congruence` or `:LS26`) |
| `:poly` *(membrane)* | Polynomial membrane | `coef`, `shape` |

> For membrane types, set `sequence_type=:p2id` when constructing `TruncatedTensorAlgebra`.

```@docs
SignatureTensors.sig
```

---

## Tensor Learning & Path Recovery

Tools for the inverse problem of recovering a path from its signature tensor. `recover`
solves the polynomial system $S = A * C$ using Gröbner bases, where $C$ is a core tensor
(axis, polynomial, spline, or membrane). The optional `algorithm=:Sch25` selects the
efficient congruence-stabilizer method that scales as $O(d^4)$ in expectation.

```@docs
SignatureTensors.recover
```

---

## Barycenters

Computation of Lie group barycenters on $G_{d,k}$, i.e. Fréchet means with respect to the
group geodesic distance. Multiple algorithms are available:

| `algorithm` | Description |
|-------------|-------------|
| *(default)* | Polynomial surjection map |
| `:geodesic` | Geodesic barycenter (Prop. 4.4, [AS25]) |
| `:AS25trunc2` | Truncation-level-2 formula (Thm. 4.11, [AS25]) |
| `:CDMSSU24aBCH` | Asymmetrized BCH series ([CDM+24]) |

```@docs
SignatureTensors.bary
```

---


## Element Operations

Standard algebraic operations on `TruncatedTensorAlgebraElem`. All operations respect the
truncated tensor algebra structure: multiplication is the shuffle/group product (truncated
at level $k$), and `exp`/`log` map between the Lie algebra and the Lie group $G_{d,k}$.

```@docs
Base.:+
Base.:-
Base.:*
Base.:^
Base.inv
Base.exp
Base.log
Base.vec
Base.:(==)
```

---

## Utilities

Lower-level helpers used internally and occasionally useful for advanced workflows.
`applyMatrixToTTA` computes matrix–tensor congruence $A * T$. `tensor_to_matrix` flattens
a graded tensor component to a matrix. `graded_component` extracts a single level from a
truncated tensor algebra element. 

```@docs
SignatureTensors.applyMatrixToTTA
SignatureTensors.tensor_to_matrix
SignatureTensors.graded_component
SignatureTensors.mode_product
```
