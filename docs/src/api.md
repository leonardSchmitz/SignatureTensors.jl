# API Reference

```@docs
SignatureTensors.SignatureTensors
```

## Types
```@docs
SignatureTensors.TruncatedTensorAlgebra
SignatureTensors.FreeTruncSigAlgMultiv
SignatureTensors.FreeTruncSigAlgMultivElem
```

## Core Functions
```@docs
SignatureTensors.sig
SignatureTensors.mode_product
SignatureTensors.free_trunc_sig_alg_multiv
SignatureTensors.gens_in_shape
SignatureTensors.graded_component
SignatureTensors.polynomial_ring_sig_transform
SignatureTensors.free_sig_from_sample
```

## Barycenter & Moments
```@docs
SignatureTensors.free_barycenter_2samples
SignatureTensors.free_sig_bary
SignatureTensors.bary_defining_polynomial_system
SignatureTensors.moment_path_level
SignatureTensors.moment_membrane_p2
SignatureTensors.moment_membrane_p2id
SignatureTensors.recover
```

## Element Operations
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
AbstractAlgebra.gens
```

## Utilities
```@docs
SignatureTensors.applyMatrixToTTA
SignatureTensors.tensor_to_matrix
SignatureTensors.sig2parPoly
SignatureTensors.bary
```