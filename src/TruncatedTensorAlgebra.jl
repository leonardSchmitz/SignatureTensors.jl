export TruncatedTensorAlgebra,
       TruncatedTensorAlgebraElem,
       truncation_level,
       base_dimension,
       sequence_type,
       base_algebra,
       tensor_sequence,
       zero,
       one,
       bary,
       bary_all_but_last_samples_fixed_inverse, 
       sig
       #matrix_tensorAlg_congruence_TA,

       #sigAxis_ClosedForm,  # sr
       #sigAxis_p2id_ClosedForm,  # sr
       #sigAxis_p2_ClosedForm,  # sr
       #sigAxis_Chen,  # sr
       #sigAxis_p2id_Chen,  # sr
       #sigAxis_p2_Chen,  # sr

       #Moment functions, 
       #moment_path_level,  # sr
       #moment_membrane,  # sr
       #moment_membrane_p2id,  # sr
       #moment_membrane_p2,  # sr

      # Tensor algebra operations
       #applyMatrixToTTA, 
       #sig2parPoly

"""
    TruncatedTensorAlgebra{R}

Structure that stores a **truncated signature** of a rough path or a
two-parameter object (membrane).

Concrete signatures are stored as elements of this algebra via
[`TruncatedTensorAlgebraElem`](@ref).

# Fields
- `base_algebra::R`  
  Base algebra over which the signature is defined.

- `base_dimension::Int`  
  Dimension `d` of the underlying path or membrane.

- `truncation_level::Int`  
  Truncation level `k` of the signature.

- `sequence_type::Symbol`  
  Type of signature:
  - `:iis`   — iterated-integrals signature of a rough path (default),
  - `:p2id`  — two-parameter signature for membranes

# Example
```julia-repl
julia> A = TruncatedTensorAlgebra(QQ, 2, 3; sequence_type = :iis)
TruncatedTensorAlgebra(QQ, 2, 3, :iis)
```

See also [`TruncatedTensorAlgebraElem`](@ref).
"""
struct TruncatedTensorAlgebra{R}
    base_algebra::R
    base_dimension::Int
    truncation_level::Int
    sequence_type::Symbol
end

"""
    TruncatedTensorAlgebraElem{R,E}

Concrete **truncated signature** of a rough path or membrane.

This object stores the actual tensor coefficients of the signature up to
a fixed truncation level.

# Type Parameters
- `R`: base algebra of the parent signature object.
- `E`: type of the coefficients at each tensor level.

# Fields
- `parent::TruncatedTensorAlgebra{R}`  
  Signature metadata (dimension, truncation level, type).

- `elem::Vector{Array{E}}`  
  Sequence of tensors representing the signature:
    - `elem[:]` level 1  (vector),
    - `elem[:,:]`: level 2,
    - `elem[:,:,:, ..., :]`: level k,

See also [`TruncatedTensorAlgebra`](@ref TruncatedTensorAlgebra)
"""
struct TruncatedTensorAlgebraElem{R,E}
    parent::TruncatedTensorAlgebra{R}
    elem::Vector{Array{E}}
end


"""
    truncation_level(F::TruncatedTensorAlgebra)

Return the truncation level `k` of the truncated tensor algebra `F`.

# Description
This corresponds to the highest tensor level stored in the truncated
signature:
```math
    S^{\\leq k}(X) = (1, S^1(X), ..., S^k(X))
```
# Returns
- `Int`: truncation level `k`.
"""
truncation_level(F::TruncatedTensorAlgebra) = F.truncation_level


"""
    base_dimension(F::TruncatedTensorAlgebra)

Return the base dimension `d` of the truncated tensor algebra `F`.

# Description
This is the dimension of the underlying path or membrane from which
the signature is computed.

# Returns
- `Int`: dimension `d`.
"""
base_dimension(F::TruncatedTensorAlgebra) = F.base_dimension


"""
    base_algebra(F::TruncatedTensorAlgebra)

Return the base algebra of the truncated tensor algebra `F`.

# Description
This specifies the algebra over which tensor coefficients of the
signature are defined.

# Returns
- `R`: base algebra.
"""
base_algebra(F::TruncatedTensorAlgebra) = F.base_algebra


"""
    sequence_type(F::TruncatedTensorAlgebra)

Return the sequence type of the truncated tensor algebra `F`.

# Description
Indicates the type of signature stored in the algebra:
- `:iis`   — iterated-integrals signature (rough paths),
- `:p2id`  — two-parameter signature (membranes).

# Returns
- `Symbol`: sequence type.
"""
sequence_type(F::TruncatedTensorAlgebra) = F.sequence_type


"""
    parent(a::TruncatedTensorAlgebraElem)

Return the parent truncated tensor algebra of the element `a`.

# Description
The parent contains the structural metadata of the signature,
including dimension, truncation level, and sequence type.

# Returns
- `TruncatedTensorAlgebra`: parent algebra.
"""
Base.parent(a::TruncatedTensorAlgebraElem) = a.parent


"""
    tensor_sequence(a::TruncatedTensorAlgebraElem)

Return the tensor sequence representing the truncated signature.

# Description
The result is a vector of tensors:
- index `1`: level 0 tensor (escalar),
- index `2`: level 1 tensor (vector),
- index `3`: level 2 tensor (matrix),
- ...
- index `k`: level `k-1` tensor.

Each entry is an array of order equal to its level.

# Returns
- `Vector{Array{E}}`: tensor sequence.
"""
tensor_sequence(a::TruncatedTensorAlgebraElem) = a.elem


function TruncatedTensorAlgebra(R, d::Int, k::Int; sequence_type::Symbol=:iis)

    d >= 0 || error("ambient dimension must be >= 0")
    k >= 0 || error("truncation level must be >= 0")

    if !(sequence_type in (:iis, :p2id, :p2))
        error("sequence_type must be one of :iis, :p2id, :p2")
    end

    A = TruncatedTensorAlgebra{typeof(R)}(R, d, k, sequence_type)

    return A
end


function _C0_TA(_k::Int, _order::Int, _alg, is_one::Bool)
    if _k == 0
        return fill(is_one ? one(_alg) : zero(_alg), ())
    else
        #return zeros(_alg, ntuple(_ -> _order, _k)...)
       return fill(zero(_alg), ntuple(_ -> _order, _k)...) 
    end
end

function _C0_TA_p2(_k::Int, _order::Int, _alg, is_one::Bool)
    if _k == 0
        return fill(is_one ? one(_alg) : zero(_alg), ())
    elseif _k == 1
        # shape (d,) — solo 1 permutación en Σ_1, no añade dimensión útil
        # pero para consistencia: shape (d, 1)
        return fill(zero(_alg), _order, 1)
    else
        # shape (d, d, ..., d, k!) con k dimensiones d y 1 dimensión k!
        n_perms = factorial(_k)
        return fill(zero(_alg), ntuple(_ -> _order, _k)..., n_perms)
    end
end

function _C0_seq_TA_p2(_trunc_level::Int, _order::Int, _alg, is_one::Bool)
    return [_C0_TA_p2(_k, _order, _alg, is_one) for _k in 0:_trunc_level]
end

function tensor_sequence_constructor(A::TruncatedTensorAlgebra; is_one=true)
    R   = A.base_algebra
    d   = A.amb_dim
    k   = A.trunc_level
    seq = A.sequence_type

    elems = Vector{Any}(undef, k+1)

    # Level 0 
    elems[1] = fill(is_one ? one(R) : zero(R), ())

    # iis
    if seq == :iis
        for n in 1:k
            elems[n+1] = fill(zero(R), ntuple(_ -> d, n)...)
        end
        return elems
    end

    # p2id
    if seq == :p2id
        for n in 1:k
            dims = ntuple(_ -> d, n)
            elems[n+1] = Array{Any}(undef, dims...)
        end
        return elems
    end

    # p2
    if seq == :p2
        for n in 1:k
            dims = (ntuple(_ -> d, n)..., factorial(n))
            elems[n+1] = Array{Any}(undef, dims...)
        end
        return elems
    end

    error("Unknown sequence_type = $seq")
end


function sig_segment_CF(T::TruncatedTensorAlgebra{R}, v::Vector{E}) where {R,E}
    k_max = truncation_level(T)
    d     = base_dimension(T)
    alg   = base_algebra(T)

    R_tensor = base_algebra(T)
    R_matrix = parent(v[1])   
    Rnew = common_ring(R_tensor, R_matrix)  
    d_new = length(v)
    Tnew = TruncatedTensorAlgebra(Rnew, d_new, k_max, :iis)


    seq = Vector{Array{typeof(one(Rnew))}}(undef, k_max + 1)

    for k in 0:k_max
        if k == 0
            seq[1] = fill(one(Rnew))   
        else

            seq[k+1] = _sig_level_tensor(v, k, Rnew)
        end
    end


    return TruncatedTensorAlgebraElem(Tnew, seq)
end


function _sig_level_tensor(v::Vector{E}, k::Int, Rnew) where E
    d = length(v)
    fact = factorial(k)
    Tcoeff = Rnew

    if k == 1

        return [Tcoeff(vi) for vi in v]
    else

        dims = ntuple(_ -> d, k)
        T = Array{typeof(Tcoeff(one(v[1])))}(undef, dims...)


        for I in Iterators.product(ntuple(_ -> 1:d, k)...)
            T[I...] = prod(Tcoeff(v[i]) for i in I) / Tcoeff(fact)
        end

        return T
    end
end

function _C0_seq_TA(_trunc_level::Int, _order::Int, _alg, is_one::Bool)
    return [_C0_TA(_k, _order, _alg, is_one) for _k in 0:_trunc_level]
end


#### Base Constructor

"""
    Base.zero(T::TruncatedTensorAlgebra)

Return the **zero truncated signature** associated with the truncated tensor
algebra `T`.

This function constructs a `TruncatedTensorAlgebraElem` whose tensor
coefficients are identically zero at all levels. It represents the additive
identity in the space of truncated signatures.

# Arguments
- `T::TruncatedTensorAlgebra`  

# Returns
- `TruncatedTensorAlgebraElem`  
  An element of `T` whose tensor coefficients are zero at every level.

# Example
```julia-repl
using Oscar
T = TruncatedTensorAlgebra(QQ, 3, 2; sequence_type = :iis)
z = zero(T)
```
"""
function Base.zero(T::TruncatedTensorAlgebra{R}) where R

    k = truncation_level(T)
    d = base_dimension(T)
    RA = base_algebra(T)   
    seq = T.sequence_type
    E = typeof(one(RA))

    elem = Vector{Array{E}}(undef, k + 1)


    elem[1] = fill(zero(RA), ())   


    if seq == :iis
        for n in 1:k


            elem[n + 1] = fill(zero(RA), ntuple(_ -> d, n)...)
        end

    elseif seq == :p2id
        for n in 1:k
            if n == 1
                dims = (d,)
            else
                dims = ntuple(_ -> d, n)
            end
            elem[n + 1] = fill(zero(RA), dims...)

        end

    elseif seq == :p2
        for n in 1:k
            if n == 1
                dims = (d,)
            else
                dims = (ntuple(_ -> d, n)..., factorial(n))
            end
            elem[n + 1] = fill(zero(RA), dims...)
        end

    else
        throw(ArgumentError("zero(T) not implemented for sequence_type = $seq"))
    end

    return TruncatedTensorAlgebraElem{R,E}(T, elem)
end


"""
    one(T::TruncatedTensorAlgebra)

Return the **identity truncated signature** associated with the truncated tensor
algebra `T`.


# Arguments
- `T::TruncatedTensorAlgebra`  

# Returns
- `TruncatedTensorAlgebraElem`  
  An element of `T` whose tensor coefficients are the identity signature.

# Example
```julia-repl
using Oscar
T = TruncatedTensorAlgebra(QQ, 3, 2; sequence_type = :iis)
z = one(T)
```
"""
function Base.one(T::TruncatedTensorAlgebra{R}) where R
    if T.sequence_type == :iis

        k = truncation_level(T)
        d = base_dimension(T)
        alg = T.base_algebra
        E = typeof(one(alg))

        return TruncatedTensorAlgebraElem{R,E}(T, _C0_seq_TA(k, d, alg, true))
    elseif T.sequence_type == :p2id

        k = truncation_level(T)
        d = base_dimension(T)
        alg = T.base_algebra
        E = typeof(one(alg))

        return TruncatedTensorAlgebraElem{R,E}(T, _C0_seq_TA(k, d, alg, true))
    elseif T.sequence_type == :p2
        k = truncation_level(T)
        d = base_dimension(T)
        alg = T.base_algebra
        E = typeof(one(alg))
        return TruncatedTensorAlgebraElem{R,E}(T, _C0_seq_TA_p2(k, d, alg, true))
    else
        throw(ArgumentError("one(T) is only defined for sequence_type = :iis, :p2id"))
    end
end


"""
    sig(T::TruncatedTensorAlgebra, geom_type::Symbol; coef = [], shape = [], composition::Vector{Int} = Int[], regularity::Int = 0,algorithm::Symbol = :default)

Compute the **truncated signature** associated with a specified path type
inside the truncated tensor algebra `T`.

# Arguments
- `T::TruncatedTensorAlgebra`  
  Truncated tensor algebra specifying the dimension, truncation level,
  base algebra, and signature type.

- `geom_type::Symbol`  
  Type of path whose signature is to be computed. Supported values include:

  ## For `:iis` sequence type
  - `:point`  
    Signature of a constant path (unit signature: level-0 is 1, higher levels are 0).

  - `:segment`  
    Signature of a single segment; requires `coef` to specify vector.

  - `:axis`  
    Signature of a path moving along coordinate axes.  
    Optional `algorithm`:
      - `:AFS19` – Closed-form formulas [AFS2019](@cite)
      - `:Chen` – Computes using Chen’s identity

  - `:mono`  
    Signature of a monomial path.

  - `:pwln`  
    Piecewise linear path. Optional arguments:
      - `coef` – Matrix of coefficients
      - `algorithm` – `:Chen`, `:congruence`, or `:LS26` [schmitz2025efficientalgorithmtensorlearning](@cite)

  - `:pwmon`  
    Piecewise monomial path. Optional arguments:
      - `composition` – Exponents of each segment
      - `regularity` – Regularity of the path
      - `algorithm` – `:Chen` or `:ALS26` [amendola2026signaturevarietiessplines](@cite)

  - `:poly`  
    Polynomial path. Optional `algorithm`:
      - `:congruence` or `:default`
      - `:ARS26` – Alternative method [bib:ARS26](@cite)

  - `:spline`  
    Spline path with optional `coef`, `composition`, and `regularity`

  ## For `:p2id` sequence type
  - `:point` – Constant path
  - `:mono` – Monomial membrane
  - `:axis` – Axis-aligned membrane, `algorithm` can be `:AFS19`[AFS2019](@cite) or `:Chen`
  - `:poly` – Polynomial membrane; if `coef ` is a matrix, it is interpreted as coefficients of a 2-parameter polynomial; otherwise, it is interpreted as a vector of coefficients for the monomial basis.
  - `:pwbln` – Piecewise bilinear path; `algorithm` can be `:congruence` or `:LS26` [amendola2026signaturevarietiessplines](@cite)

# Notes
- The function automatically dispatches based on `sequence_type(T)`:
  - `:iis` – Indexed iterated sequences (typical rough path)
  - `:p2id` – 2-parameter (id) signature
  - `:p2` – Currently not supported
- `coef` and `shape` provide the data describing the path when needed.
- `composition` and `regularity` are used for piecewise monomial or spline paths.
- Throws `ArgumentError` if unsupported combination of arguments is provided.
"""
function sig(T::TruncatedTensorAlgebra{R}, geom_type::Symbol; coef=[], shape=[], composition::Vector{Int}=Int[], regularity::Int=0,  algorithm::Symbol=:default) where R
    seq_type = sequence_type(T)

    if seq_type == :iis
        if geom_type == :point && coef == [] && algorithm == :default
            return one(T)
        elseif geom_type == :segment && (algorithm == :default || algorithm == :congruence)
            return sig_segment_TA(T, Array(coef))
        elseif geom_type == :segment && algorithm == :closedform 
            return sig_segment_CF(T, Array(coef))
        elseif geom_type == :axis && coef == [] && (algorithm == :default || algorithm == :AFS19)
            return sigAxis_TA_ClosedForm(T)
        elseif geom_type == :axis && coef == [] && algorithm == :Chen
            return sig_axis_TA(T)
        elseif geom_type == :mono && coef == [] && algorithm == :default
            return sig_mono_TA(T)
        elseif geom_type == :pwln && algorithm == :congruence
            return sig_pwln_TA_Congruence(T, Array(coef))
        elseif geom_type == :pwln && (algorithm == :Chen || algorithm == :default)
            return sig_pwln_TA_chen(T, Array(coef))
        elseif geom_type == :pwln && algorithm == :LS26
            return sig_pwln_TA_LS26(T,Array(coef))
        elseif geom_type == :pwmon && algorithm == :Chen
            return sig_pw_mono_chen(T, composition, regularity)
        elseif geom_type == :pwmon && (algorithm == :ALS26 || algorithm == :default)
            return sig_pw_mono_ALS26(T, composition, regularity)
        elseif geom_type == :poly && ( algorithm == :congruence || algorithm == :default )
            return sig_poly_TA(T,Array(coef))
        elseif geom_type == :poly && ( algorithm == :ARS26 )
            return sig_poly_TA_ARS(T,Array(coef))
        elseif geom_type == :spline 
            return sig_spline(T,coef,composition,regularity) 
        else
            throw(ArgumentError("sig not supported for given arguments"))
        end

    elseif seq_type == :p2id
        if geom_type == :point && coef == [] && algorithm == :default
            return one(T)
        elseif geom_type == :mono && coef == [] && algorithm == :default
            return moment_membrane_p2id(T, shape[1], shape[2])
        elseif geom_type == :axis && coef == [] && (algorithm == :default || algorithm == :AFS19)
            return sigAxis_p2id_ClosedForm(T, shape[1], shape[2])
        elseif geom_type == :axis && coef == [] && algorithm == :Chen
            return sigAxis_p2id_Chen(T, shape[1], shape[2])
        elseif geom_type == :poly && algorithm == :default
            if ndims(coef) == 2
                return sig2parPoly_fromMatrix(T, coef, shape[1], shape[2])
            else
                return sig2parPoly(T, coef)
            end
        elseif geom_type == :pwbln && (algorithm == :default || algorithm == :congruence)
            if ndims(coef) == 2
                return sig_pwbln_p2id_Congruence(T, coef, shape[1], shape[2])
            else
                return sig_pwbln_p2id_Congruence_fromTensor(T, coef, size(coef,1), size(coef,2))
            end
        elseif geom_type == :pwbln && algorithm == :LS26
            if ndims(coef) == 2
                return sigPiecewiseBilinear_TA(T, coef, shape)
            else
               return sigPiecewiseBilinear_fromTensor_TA(T, coef, [size(coef,1), size(coef,2)])
            end
        else
            throw(ArgumentError("sig not supported for given arguments"))
        end
    elseif seq_type == :p2
        if geom_type == :point && coef == [] && algorithm == :default
            return one(T)
        elseif geom_type == :mono && coef == [] && algorithm == :default
            return moment_membrane_p2(T, shape[1], shape[2])
        elseif geom_type == :axis && coef == [] && (algorithm == :default || algorithm == :AFS19)
            return sigAxis_p2_ClosedForm(T, shape[1], shape[2])
        elseif geom_type == :axis && coef == [] && algorithm == :Chen
            return sigAxis_p2_Chen(T, shape[1], shape[2])
        elseif geom_type == :poly && algorithm == :default
            if ndims(coef) == 2
                return sig2parPoly_fromMatrix_p2(T, coef, shape[1], shape[2])
            else
                return sig2parPoly_p2(T, coef)
            end
        elseif geom_type == :pwbln && (algorithm == :default || algorithm == :congruence)
            if ndims(coef) == 2
                return sig_pwbln_p2_Congruence(T, coef, shape[1], shape[2])
            else
                return sig_pwbln_p2_Congruence_fromTensor(T, coef, size(coef,1), size(coef,2))
            end
        elseif geom_type == :pwbln && algorithm == :LS26
            if ndims(coef) == 2
                return sigPiecewiseBilinear_TA_p2(T, coef, shape)
            else
               return sigPiecewiseBilinear_fromTensor_TA_p2(T, coef, [size(coef,1), size(coef,2)])
            end
        else 
            throw(ArgumentError("sig not supported for given arguments"))
        end

    else
        throw(ArgumentError("sig not supported for given arguments"))
    end
end



function Base.show(io::IO, x::TruncatedTensorAlgebraElem)
    for (i, t) in enumerate(x.elem)
        show(io, MIME("text/plain"), t)
        if i < length(x.elem)
            println(io, "\noplus")
        end
    end
end

function Oscar.evaluate(
    p::FreeTruncSigAlgMultivElem{T},
    bs::Vector{TruncatedTensorAlgebraElem{R,E}}
) where {T,R,E}

    A = parent(bs[1])               
    k = truncation_level(A)         

    res_seq = [evaluate_to_tensor_TA(p, bs, i) for i in 0:k]

    return TruncatedTensorAlgebraElem(A, res_seq)
end

function _CmonoTA(_k::Int, _order::Int, R0)
    if _k == 0
        return fill(one(R0), ())
    end

    E = typeof(one(R0))

    res = Array{E}(undef, ntuple(_ -> _order, _k)...)

    for idx in CartesianIndices(res)
        word = Tuple(idx)
        coeff = prod(word)
        expo  = prod(cumsum(word))
        res[idx] = QQ(coeff, expo) * one(R0)
    end

    return res
end


function _Cmono_seqTA(_trunc_level::Int, _order::Int, R0)
    return [_CmonoTA(_k, _order, R0) for _k in 0:_trunc_level]
end

function sig_mono_TA(T::TruncatedTensorAlgebra{R}) where R
    if T.sequence_type != :iis
        throw(ArgumentError("sig_mono only defined for sequence_type = :iis"))
    end

    k = truncation_level(T)
    d = base_dimension(T)
    R0 = base_algebra(T)
    seq = _Cmono_seqTA(k, d, R0)

    E = eltype(seq[1])  
    return TruncatedTensorAlgebraElem{R, E}(T, seq)
end

#helpters for Pw polynomial paths

function embedding_matrix(m::Vector{Int},i::Int)
  M = sum(m)
  A = zero_matrix(QQ,M,m[i])
  A[sum(m[1:i-1])+1:sum(m[1:i]),:] = identity_matrix(QQ,m[i])
  return A
end

function nextBlock(A, r::Int, mi)
    nrows, ncols = size(A)
    vecs_qqmatrix = zero_matrix(QQ, ncols, r)               
    vecs = [vecs_qqmatrix[i,j] for i in 1:ncols, j in 1:r]  
    for i in 1:r
        for j in 1:ncols
            vecs[j, i] = binomial(j, i)
        end
    end
    return matrix(QQ,Matrix(A) * Matrix(vecs))
end

function coreSplineTrafo(m::Vector{Int}, r::Int) 
    total_dim = sum(m)
    if r == 0
        return identity_matrix(QQ,total_dim)
    end
    zz = 1
    crb = identity_matrix(QQ,m[1])
    B = crb
    while zz < length(m)
        nr = size(B, 2)
        start_col = nr - m[zz] + 1
        end_col   = nr
        subB = B[:, start_col:end_col]
        nb = nextBlock(subB, r, zz)
        idpart = identity_matrix(QQ,m[zz+1] - r)
        B = block_diagonal_matrix([hcat(B, nb), idpart])
        zz += 1
    end
    return B
end

function sig_pw_mono_chen(TTS::TruncatedTensorAlgebra,m,r) 
  k = truncation_level(TTS)
  R = base_algebra(TTS)
  d = base_dimension(TTS)
  if d != sum(m) - r*(length(m) -1)
    error("m must be a composition of the ambient dimension")
  end
  sigs = [Array(embedding_matrix(m,i))*sig(TruncatedTensorAlgebra(R,m[i],k),:mono) for i in (1:length(m))]
  res = prod(sigs)
  if r == 0
    return res
  else
    return Array(coreSplineTrafo(m,r))*res
  end
end

function adapted_word(w::Vector{Int}, m::Vector{Int})
    ell = length(m)
    offsets = cumsum([0; m[1:end-1]])  
    blocks = [Int[] for _ in 1:ell]
    current = 1
    for x in w
        while current <= ell && x > offsets[current] + m[current]
            current += 1
        end
        if current > ell || x ≤ offsets[current]
            return false,[]
        end
        push!(blocks[current], x)
    end
    return true,[blocks[i] .- offsets[i] for i in 1:ell]
end

function _Cpwpoly(_k::Int, m::Vector{Int}, _R)
   M = sum(m)
   res = fill(zero(_R), M*ones(Int,_k)...)
   for idx in CartesianIndices(res)
       if _k ==0 
         res[idx] = one(_R)
       else
         b,vs = adapted_word(collect(Tuple(idx)),m)
         if b
           res[idx] = prod([QQ(prod(v),prod(cumsum(v))) for v in vs])*one(_R)
         end
       end 
   end
   return res
end

function _Cpwpoly_seq(_trunc_level::Int, m::Vector{Int}, _R)
   Cmono = [_Cpwpoly(i,m, _R) for i in (0:_trunc_level)]
   return Cmono
end

function sig_pw_mono_ALS26(TTS::TruncatedTensorAlgebra,m::Vector{Int},r)
  d = base_dimension(TTS)
  if d != sum(m) - r*(length(m) -1)
    error("m must be a composition of the ambient dimension")
  end
  k = truncation_level(TTS) 
  R = base_algebra(TTS)
  TM = TruncatedTensorAlgebra(R,sum(m),k)
  res = TruncatedTensorAlgebraElem(TM,_Cpwpoly_seq(k,m,R))
  if r == 0
    return res
  else
    return Array(coreSplineTrafo(m,r))*res
  end
end


function sig_spline(TTS::TruncatedTensorAlgebra{R},coeffs::AbstractMatrix{E},m::Vector{Int},r) where {R,E}
  d = base_dimension(TTS)
  @assert size(coeffs,1) == d "Dimensions mismatch"
  ms=size(coeffs,2)
           
  if ms != sum(m) - r*(length(m) -1)
    error("m must be a composition of size(coefffs,2)")
  end
  k = truncation_level(TTS) 
  R0 = base_algebra(TTS)
  T2=TruncatedTensorAlgebra(R0,ms,k)
  TM=sig_pw_mono_chen(T2,m,r)
  spline_path = matrix_tensorAlg_congruence_TA(coeffs, TM)
  return spline_path
end




# Base functions
"""
    Base.getindex(x::TruncatedTensorAlgebraElem, w...)
Return a tensor element of `x` at the level determined by the number of indices `w...`.

# Description
- Access tensors at different **levels** of a truncated tensor algebra element.
- The number of indices `w` determines the **level**:
    - `S[:]` → level 1 (vector)
    - `S[:,:]` → level 2 (matrix)
    - `S[:,:,:]` → level 3 (tensor)
    - and so on for level `k`.

# Arguments
- `x::TruncatedTensorAlgebraElem` : element of a truncated tensor algebra.  
- `w...` : indices to access the tensor at the given level.

# Returns
- A tensor of order `k`, indexed by `w...`.

# Example
```julia
using Oscar
d, k = 2, 3
T = TruncatedTensorAlgebra(QQ, d, k)

# Signature of the canonical axis path
S=sig(T, :axis)

S[1]       # level 1
S[:]       # level 1 full
S[:, :]    # level 2
S[:, :, :] # level 3
```
"""
function Base.getindex(x::TruncatedTensorAlgebraElem, w...)
    k=length(w)
    b=parent(x)
    typeT=b.sequence_type
    if k < 0
        throw(BoundsError(x, k))
    end
    
    if k > length(x.elem)
        throw(BoundsError(x, k))
    end
    if typeT==:iis 
        temp= x.elem[k+1 ]
    elseif typeT==:p2
        if k==1
            temp=x.elem[k+1]
        else
            temp=x.elem[k]
        end
    elseif typeT==:p2id
        if k==1
            temp=x.elem[k+1]
        else
            temp=x.elem[k+1]
        end
    else 
        error("seq must be one of :iis, :p2id, :p2")
    end
    
    return temp[w...]
end


"""
    vec(x::TruncatedTensorAlgebraElem) -> Vector

Flatten a `TruncatedTensorAlgebraElem` into a single vector by concatenating
the vectorizations of each tensor in its sequence.

# Arguments
- `x::TruncatedTensorAlgebraElem`: Element of a truncated tensor algebra.

# Returns
A `Vector` with all entries of the tensor sequence laid out contiguously.

# Example
```julia
using Oscar
d, k = 2, 3
T = TruncatedTensorAlgebra(QQ, d, k)

# Signature of the canonical axis path
S=sig(T, :axis)


v = vec(S)
```
"""
function Base.vec(x::TruncatedTensorAlgebraElem)
    return vcat([vec(xi) for xi in x.elem]...)
end



"""
    ==(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem) -> Bool

Test equality of two elements of a truncated tensor algebra.

Two elements are equal if they share the same parent algebra and their tensor
sequences are equal. Only defined for `sequence_type` ∈ `{:iis, :p2id, :p2}`.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Left-hand side.
- `b::TruncatedTensorAlgebraElem`: Right-hand side.

# Returns
`true` if same parent and same tensor sequence, `false` otherwise.

# Throws
- `ArgumentError` if either element has an unsupported `sequence_type`, or if
  they have different sequence types.

# Example
```julia
using Oscar
d, k = 2, 3
T = TruncatedTensorAlgebra(QQ, d, k)

S1=sig(T, :axis)
S2=sig(T, :axis)

S1 == S2
```
"""
function Base.:(==)(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis && parent(b).sequence_type == :iis
        return parent(a) == parent(b) && tensor_sequence(a) == tensor_sequence(b)
    elseif parent(a).sequence_type == :p2id && parent(b).sequence_type == :p2id
        return parent(a) == parent(b) && tensor_sequence(a) == tensor_sequence(b)
    elseif parent(a).sequence_type == :p2 && parent(b).sequence_type == :p2
        return parent(a) == parent(b) && tensor_sequence(a) == tensor_sequence(b)
    else
        throw(ArgumentError("== only defined for sequence_type == :iis, :p2id, or :p2"))
    end
end



# =========================
# Addition and Subtraction
# =========================

"""
    Base.:+(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)

Element-wise addition of two elements of a truncated tensor algebra.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Left-hand side.
- `b::TruncatedTensorAlgebraElem`: Right-hand side.

# Returns
A new `TruncatedTensorAlgebraElem` whose tensor sequence is the level-wise
sum of those of `a` and `b`.

# Throws
- `ArgumentError` if the `sequence_type` of `a` or `b` is not `:iis`, `:p2id`, or `:p2`,
  or if they have different sequence types.

# Notes
- Both elements must share the same parent algebra.
- Addition is defined for `sequence_type` ∈ `{:iis, :p2id, :p2}`.
"""
function Base.:+(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis && parent(b).sequence_type == :iis
        A = parent(a)
        res_seq = tensor_sequence(a) + tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(a).sequence_type == :p2id && parent(b).sequence_type == :p2id
        A = parent(a)
        res_seq = tensor_sequence(a) + tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(a).sequence_type == :p2 && parent(b).sequence_type == :p2
        A = parent(a)
        res_seq = tensor_sequence(a) + tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    else
        throw(ArgumentError("+ only defined for sequence_type == :iis, :p2id, or :p2"))
    end
end


"""
    Base.:-(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)

Element-wise subtraction of two elements of a truncated tensor algebra.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Left-hand side.
- `b::TruncatedTensorAlgebraElem`: Right-hand side.

# Returns
A new `TruncatedTensorAlgebraElem` whose tensor sequence is the level-wise
difference of those of `a` and `b`.

# Throws
- `ArgumentError` if the `sequence_type` of `a` or `b` is not `:iis`, `:p2id`, or `:p2`,
  or if they have different sequence types.

# Notes
- Both elements must share the same parent algebra.
- Subtraction is defined for `sequence_type` ∈ `{:iis, :p2id, :p2}`.
- Analogous to `+`, see [`Base.:+`](@ref).
"""
function Base.:-(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis && parent(b).sequence_type == :iis
        A = parent(a)
        res_seq = tensor_sequence(a) - tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(a).sequence_type == :p2id && parent(b).sequence_type == :p2id
        A = parent(a)
        res_seq = tensor_sequence(a) - tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(a).sequence_type == :p2 && parent(b).sequence_type == :p2
        A = parent(a)
        res_seq = tensor_sequence(a) - tensor_sequence(b)
        return TruncatedTensorAlgebraElem(A, res_seq)
    else
        throw(ArgumentError("- only defined for sequence_type == :iis, :p2id, or :p2"))
    end
end




# =========================
# Multiplication (Chen / Congruence / Scalar)
# =========================

"""
    Base.:*(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)

Chen product (concatenation product) of two elements in a truncated tensor algebra,
implementing Chen's identity for iterated integrals.

For two signatures `a` and `b` of a path, the product at level `i` is given by:
```math
(a \\cdot b)_i = a_i + b_i + \\sum_{j=1}^{i-1} a_j \\otimes b_{i-j}
```

where `⊗` denotes the tensor product of arrays (`concatenate_tensors`).

# Arguments
- `a::TruncatedTensorAlgebraElem{R,E}`: Left-hand side element.
- `b::TruncatedTensorAlgebraElem{R,E}`: Right-hand side element.

# Returns
A new `TruncatedTensorAlgebraElem` representing the Chen product of `a` and `b`,
truncated at the same level `k` as the parent algebra.

# Throws
- `ArgumentError` if the `sequence_type` of `a` and `b` is not a matching pair of
  `:iis` or `:p2id`. The type `:p2` is not supported for this operation.

# Notes
- Both elements must share the same parent algebra and the same `sequence_type`.
- The level-0 component is inherited from `a` (as expected for the Chen product).
- The level-1 component is the sum `a₁ + b₁`.
- For levels `i ≥ 2`, the cross terms `aⱼ ⊗ b_{i-j}` are accumulated via
  `concatenate_tensors`.
- See also: [`Base.:+`](@ref), [`Base.:-`](@ref).
"""
function Base.:*(a::TruncatedTensorAlgebraElem{R,E}, 
                 b::TruncatedTensorAlgebraElem{R,E}) where {R,E}

    seq_type_a = parent(a).sequence_type
    seq_type_b = parent(b).sequence_type

    if seq_type_a == :iis && seq_type_b == :iis
        A = parent(a)
        k = truncation_level(A)
        res_seq_a = tensor_sequence(a)
        res_seq_b = tensor_sequence(b)

        res_seq_new = Vector{Array{E}}(undef, k+1)
        res_seq_new[1] = res_seq_a[1]
        res_seq_new[2] = res_seq_a[2] + res_seq_b[2]

        if k >= 2
            for i in 2:k
                temp = res_seq_a[i+1] + res_seq_b[i+1]
                for j in 1:i-1
                    temp += concatenate_tensors(res_seq_a[j+1], res_seq_b[i-j+1])
                end
                res_seq_new[i+1] = temp
            end
        end

        return TruncatedTensorAlgebraElem(A, res_seq_new)

    elseif seq_type_a == :p2id && seq_type_b == :p2id
        A = parent(a)
        k = truncation_level(A)
        res_seq_a = tensor_sequence(a)
        res_seq_b = tensor_sequence(b)

        res_seq_new = Vector{Array{E}}(undef, k+1)
        res_seq_new[1] = res_seq_a[1]
        res_seq_new[2] = res_seq_a[2] + res_seq_b[2]
        
        if k >= 2
            for i in 2:k
                temp = res_seq_a[i+1] + res_seq_b[i+1]
                for j in 1:i-1
                    temp += concatenate_tensors(res_seq_a[j+1], res_seq_b[i-j+1])
                end
                res_seq_new[i+1] = temp
            end
        end

        return TruncatedTensorAlgebraElem(A, res_seq_new)

    else
        throw(ArgumentError("Chen product only defined for matching sequence_type (:iis or :p2id)"))
    end
end

##################################################################################
#.    Previous implementation using free algebra evaluation.  Don't delete yet
################################################################################


#function Base.:*(a::TruncatedTensorAlgebraElem{R,E}, 
#                 b::TruncatedTensorAlgebraElem{R,E}) where {R,E}
#    if parent(a).sequence_type == :iis && parent(b).sequence_type == :iis
#      A = parent(a)
#      k = truncation_level(A)

#      F, s = free_trunc_sig_alg_multiv(k, 2)
#      chen = prod([free_sig_from_sample(i, F) for i in 1:2])
#      return evaluate(chen, [a, b])
#    elseif parent(a).sequence_type == :p2id && parent(b).sequence_type == :p2id
#      A = parent(a)
#      k = truncation_level(A)

#      F, s = free_trunc_sig_alg_multiv(k, 2)
#      chen = prod([free_sig_from_sample(i, F) for i in 1:2])
#      return evaluate(chen, [a, b])
#    else
#        throw(ArgumentError("* only defined for sequence_type == :iis, :p2id, or :p2"))
#    end
#end


#function Base.:*(a::TruncatedTensorAlgebraElem, b::TruncatedTensorAlgebraElem)
#    if parent(a).sequence_type == :iis && parent(b).sequence_type == :iis
#        A = parent(a)
#        k = truncation_level(A)
#        F, s = free_trunc_sig_alg_multiv(k, 2)
#        chen = prod([free_sig_from_sample(i, F) for i in 1:2])
#        return evaluate(chen, [a, b])
#    else
#        throw(ArgumentError("* only defined for sequence_type == :iis"))
#    end
#end

"""
    Base.:*(matrix::AbstractMatrix, x::TruncatedTensorAlgebraElem)

Matrix-tensor congruence multiplication.

- Only defined for `sequence_type == :iis`.
- Left-multiplies a truncated signature by a matrix using a tensor congruence
  operation.
"""
function Base.:*(matrix::AbstractMatrix, x::TruncatedTensorAlgebraElem)
    T = parent(x)
    if T.sequence_type == :iis
        return matrix_tensorAlg_congruence_TA(matrix, x)
    elseif  T.sequence_type == :p2id
        return applyMatrixToTTA(matrix, x)
    elseif  T.sequence_type == :p2
        return applyMatrixToTTA(matrix, x)
    else
        throw(ArgumentError("matrix * TruncatedTensorAlgebraElem ony defined for sequence_type = :iis, :p2id, :p2"))
    end
end


"""
    Base.:*(a::Number, b::TruncatedTensorAlgebraElem)

Scalar multiplication of a truncated signature.

- Only defined for `sequence_type == :iis`.
- Multiplies each tensor level by the scalar `a`.
- Supports numbers (`Number`) or algebra elements.
- Returns a new `TruncatedTensorAlgebraElem`.
"""
function Base.:*(a::FieldElem, b::TruncatedTensorAlgebraElem)
    if parent(b).sequence_type == :iis
        A = parent(b)
        k = truncation_level(A)
        res_seq = tensor_sequence(b)
        res_seq[1][] = a * res_seq[1][]        # Level 0
        for j in 2:k+1
            res_seq[j] = a .* res_seq[j]
        end
        return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(b).sequence_type == :p2id
        A = parent(b)
        k = truncation_level(A)
        res_seq = tensor_sequence(b)
        res_seq[1][] = a * res_seq[1][]        # Level 0
        for j in 2:k+1
            res_seq[j] = a .* res_seq[j]
        end
        return TruncatedTensorAlgebraElem(A, res_seq)
    else
        throw(ArgumentError("scalar * only defined for sequence_type == :iis, :p2id"))
    end
end

function Base.:*(a::R, b::TruncatedTensorAlgebraElem{R,E}) where {R,E}
    if parent(b).sequence_type == :iis
       A = parent(b)
       k = truncation_level(A)
       res_seq = tensor_sequence(b)
       res_seq[1][] = a * res_seq[1][]
       for j in 2:k+1
           res_seq[j] = a .* res_seq[j]
       end
       return TruncatedTensorAlgebraElem(A, res_seq)
    elseif parent(b).sequence_type == :p2id
       A = parent(b)
       k = truncation_level(A)
       res_seq = tensor_sequence(b)
       res_seq[1][] = a * res_seq[1][]
       for j in 2:k+1
           res_seq[j] = a .* res_seq[j]
       end
       return TruncatedTensorAlgebraElem(A, res_seq)
    else
        throw(ArgumentError("scalar * only defined for sequence_type == :iis, :p2id"))
    end
end

# =========================
# Inverse and Powers
# =========================

"""
    Base.inv(a::TruncatedTensorAlgebraElem)

Multiplicative inverse (Chen inverse) of an element in a truncated tensor algebra.

The inverse is computed by lifting `a` to a free truncated signature algebra,
inverting there, and evaluating the result back at `a`.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Element to invert.

# Returns
A new `TruncatedTensorAlgebraElem` representing `a⁻¹` in the Chen product sense,
satisfying `a * inv(a) == one(parent(a))`.

# Throws
- `ArgumentError` if `sequence_type` is not `:iis` or `:p2id`.

# Notes
- Defined for `sequence_type` ∈ `{:iis, :p2id}`. Not supported for `:p2`.
- See also: [`Base.:*`](@ref), [`Base.:^`](@ref).
"""
function Base.:inv(a::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis
        A = parent(a)
        k = truncation_level(A)
        F, s = free_trunc_sig_alg_multiv(k, 1)
        p = inv(free_sig_from_sample(1, F))
        return evaluate(p, [a])
    elseif parent(a).sequence_type == :p2id
        A = parent(a)
        k = truncation_level(A)
        F, s = free_trunc_sig_alg_multiv(k, 1)
        p = inv(free_sig_from_sample(1, F))
        return evaluate(p, [a])
    else
        throw(ArgumentError("inv only defined for sequence_type == :iis, :p2id"))
    end
end


"""
    Base.:^(a::TruncatedTensorAlgebraElem, n::Int)

Exponentiation of an element in a truncated tensor algebra via iterated Chen product.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Base element.
- `n::Int`: Integer exponent (positive, negative, or zero).

# Returns
A new `TruncatedTensorAlgebraElem` representing the `n`-fold Chen product of `a`:
- `n > 0`: `a * a * ... * a` (`n` times).
- `n == 0`: the identity element `one(parent(a))`.
- `n < 0`: `inv(a) * inv(a) * ... * inv(a)` (`|n|` times).

# Throws
- `ArgumentError` if `sequence_type` is not `:iis` or `:p2id`.

```julia
using Oscar

T = TruncatedTensorAlgebra(QQ,2, 3)
T=sig(T,:axis)

T^3
```

# Notes
- Defined for `sequence_type` ∈ `{:iis, :p2id}`. Not supported for `:p2`.
- See also: [`Base.inv`](@ref), [`Base.:*`](@ref).
"""
function Base.:^(a::TruncatedTensorAlgebraElem, n::Int)
    if parent(a).sequence_type == :iis
        A = parent(a)
        res = one(A)
        temp = n >= 0 ? a : inv(a)

        for i in 1:abs(n)
            res = res * temp
        end
        return res
    elseif parent(a).sequence_type == :p2id
        A = parent(a)
        res = one(A)
        temp = n >= 0 ? a : inv(a)

        for i in 1:abs(n)
            res = res * temp
        end
        return res
    else
        throw(ArgumentError("^ only defined for sequence_type == :iis, :p2id"))
    end
end

# =========================
# Exponential and Logarithm
# =========================

"""
    Base.exp(a::TruncatedTensorAlgebraElem)

Exponential of an element in a truncated tensor algebra, computed via the
formal power series truncated at level `k`:
```math
\\exp(a) = \\sum_{n=0}^{k} \\frac{a^n}{n!}
```

The exponential is computed by lifting `a` to a free truncated signature algebra,
evaluating `exp` there, and mapping the result back.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Element to exponentiate.

# Returns
A new `TruncatedTensorAlgebraElem` representing `exp(a)`, truncated at the
same level `k` as the parent algebra.

# Throws
- `ArgumentError` if `sequence_type` is not `:iis`.

# Example
```julia
using Oscar

T = TruncatedTensorAlgebra(QQ,2, 3)

T=sig(T,:axis)

exp(T)
```
# Notes
- Only defined for `sequence_type == :iis`. Not supported for `:p2id` or `:p2`.
- The input `a` is expected to have vanishing level-0 component, as is standard
  for log-signatures (i.e., `a` lives in the Lie algebra, not the group).
- See also: [`Base.log`](@ref), [`Base.inv`](@ref), [`Base.:^`](@ref).
"""
function Base.:exp(a::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis
        A = parent(a)
        k = truncation_level(A)
        F, s = free_trunc_sig_alg_multiv(k, 1)
        p = exp(free_sig_from_sample(1, F) - one(F))
        return evaluate(p, [a])
    else
        throw(ArgumentError("exp only defined for sequence_type == :iis"))
    end
end



"""
    Base.log(a::TruncatedTensorAlgebraElem)

Logarithm of an element in a truncated tensor algebra, computed via the
formal power series truncated at level `k`:
```math
\\log(a) = \\sum_{n=1}^{k} \\frac{(-1)^{n+1}}{n} (a - 1)^n
```

The logarithm is computed by lifting `a` to a free truncated signature algebra,
evaluating `log` there, and mapping the result back.

# Arguments
- `a::TruncatedTensorAlgebraElem`: Element to take the logarithm of.

# Returns
A new `TruncatedTensorAlgebraElem` representing `log(a)`, truncated at the
same level `k` as the parent algebra. The result lives in the Lie algebra
(vanishing level-0 component).

# Throws
- `ArgumentError` if `sequence_type` is not `:iis`.

# Example
```julia
using Oscar 

T = TruncatedTensorAlgebra(QQ,2, 3)

T=sig(T,:axis)

log(T)
exp(log(T))==T
```
# Notes
- Only defined for `sequence_type == :iis`. Not supported for `:p2id` or `:p2`.
- The input `a` is expected to have level-0 component equal to `1` (i.e., `a`
  lives in the group, not the Lie algebra).
- `log` and `exp` are mutual inverses: `log(exp(x)) == x` and `exp(log(a)) == a`.
- See also: [`Base.exp`](@ref), [`Base.inv`](@ref), [`Base.:^`](@ref).
"""
function Base.:log(a::TruncatedTensorAlgebraElem)
    if parent(a).sequence_type == :iis
        A = parent(a)
        k = truncation_level(A)
        F, s = free_trunc_sig_alg_multiv(k, 1)
        p = log(free_sig_from_sample(1, F))
        return evaluate(p, [a])
    else
        throw(ArgumentError("log only defined for sequence_type == :iis"))
    end
end


### Barycenter 

"""
    bary(bs::Vector{TruncatedTensorAlgebraElem{R, E}}; algorithm::Symbol = :default) where {R, E}

Compute the **Lie group barycenter** (in the sense of Buser and Karcher) of a collection of
elements in a truncated tensor algebra, representing truncated path signatures.

The barycenter is the unique group element ``m`` satisfying the implicit equation
```math
\\sum_{i=1}^{N} \\log(m^{-1} \\cdot b_i) = 0
```

in the associated Lie algebra. This is distinct from the naive mean (pointwise average of
logarithms), as it preserves the group structure of the truncated signature space.

See also: [amendola2025learning](@cite).

# Arguments
- `bs::Vector{TruncatedTensorAlgebraElem{R,E}}`: A nonempty vector of elements from the same
  truncated tensor algebra (all must share the same parent).
- `algorithm::Symbol`: The algorithm used to compute the barycenter. Options:
  - `:default` — calls `bary_TA`, the general iterative algorithm valid for any truncation level .
  - `:geodesic` — geodesic midpoint formula `b₁ · exp(½ · log(b₁⁻¹ · b₂))`;
    only valid for exactly **2 elements**.
  - `:CDMSSU24trunc2` — specialized algorithm from  [amendola2025learning; Theorem 4.11](@cite).
    valid only for truncation level `k = 2`.
  - `:AS25trunc2` — closed-form expression from [amendola2025learning; Theorem 4.11](@cite),
    valid only for truncation level `k = 2`.

# Returns
A `TruncatedTensorAlgebraElem{R,E}` representing the barycenter of `bs`.

# Throws
- `ArgumentError` if the combination of `algorithm` and input arguments is not supported
  (e.g., `:geodesic` with more than 2 elements, or `trunc2` algorithms with `k ≠ 2`).

# Examples
```julia
# Default algorithm (any number of elements, any truncation level)
bary(bs)

# Geodesic midpoint of exactly 2 elements
bary([b1, b2]; algorithm = :geodesic)

# Closed-form algorithm at truncation level k = 2
bary(bs; algorithm = :AS25trunc2)
```

# Notes
- All elements in `bs` must belong to the same parent algebra.
- The barycenter is **not** the pointwise expectation of tensor entries; it lives in the
  free nilpotent Lie group and can be interpreted as the signature of some path.
- For `k ≥ 3`, the barycenter differs from the naive mean in the Lie algebra;
  see [amendola2025learning; Example 4.9](@cite).
"""
function bary(bs::Vector{TruncatedTensorAlgebraElem{R, E}}; 
              algorithm::Symbol = :default) where {R, E}
    k = truncation_level(parent(bs[1]))
    if algorithm == :default
        return bary_TA(bs) 
    elseif algorithm == :geodesic && length(bs) == 2
        return bs[1]*exp(QQ(1,2)*log(inv(bs[1])*bs[2])) 
    elseif algorithm == :CDMSSU24trunc2 && k==2
        return bary_2nd_trunc_TA(bs)
    elseif algorithm == :AS25trunc2 && k==2
        return bary_2nd_trunc_closedform_TA(bs) 
    else 
        throw(ArgumentError("sig not supported for given arguments")) 
    end 
end 

function bary_2samples_TA(bs::Vector{TruncatedTensorAlgebraElem{R, E}}) where {R, E}
  return bs[1]*exp(QQ(1,2)*log(inv(bs[1])*bs[2])) 
end

function bary_TA(bs::Vector{TruncatedTensorAlgebraElem{R, E}}) where {R, E}
  TTSm = parent(bs[1])
  res = one(TTSm)
  k = truncation_level(TTSm)
  N = length(bs)
  p = bary_defining_polynomial_system(k,N) 
  s = gens_in_shape(parent(p))
  y = s[:,N+1]
  for j in (1:k)
    pj = graded_component(p,j)
    res_seq = tensor_sequence(res)
    bs_new = [bs;res]
    res_seq[j+1] = tensor_sequence(evaluate(pj+y[j],bs_new))[j+1]
    res = TruncatedTensorAlgebraElem(TTSm,res_seq) 
  end 
  return res 
end

function bary_all_but_last_samples_fixed_inverse(bs::Vector{TruncatedTensorAlgebraElem{R, E}},
                                                 y::TruncatedTensorAlgebraElem{R, E}) where {R, E}
  TTSm = parent(bs[1])
  res = one(TTSm)
  k = truncation_level(TTSm)
  N = length(bs) + 1
  p = bary_defining_polynomial_system(k,N) 
  s = gens_in_shape(parent(p))
  xN = s[:,N]
  for j in (1:k)
    pj = graded_component(p,j)
    res_seq = tensor_sequence(res)
    bs_new = [[bs[1:N-1];res];y]
    res_seq[j+1] = tensor_sequence(evaluate(-QQ(N,1)*pj+xN[j],bs_new))[j+1]
    res = TruncatedTensorAlgebraElem(TTSm,res_seq) 
  end 
  return res 
end

function bary_2nd_trunc_TA(bs::Vector{TruncatedTensorAlgebraElem{R, E}}) where {R, E}
  N = length(bs)
  return exp(QQ(1,N)*sum(log.(bs))) 
end

function bary_2nd_trunc_closedform_TA(bs::Vector{TruncatedTensorAlgebraElem{R, E}}) where {R, E}
  TTSm = parent(bs[1])
  N = length(bs)
  ls = tensor_sequence.(bs)
  vec = QQ(1,N).*sum(ls[i][2] for i in (1:N))
  mat = QQ(1,N).*sum(ls[i][3] for i in (1:N)) - QQ(1,2*N).*sum(ls[i][2]*transpose(ls[i][2]) for i in (1:N)) + QQ(1,2*N^2).*sum(ls[i1][2]*transpose(ls[i2][2]) for i1 in (1:N) for i2 in (1:N))
  return TruncatedTensorAlgebraElem(TTSm,[ls[1][1],vec,mat])
end

### Ideal constuctors


function ideal_of_lyndon_entries(x::TruncatedTensorAlgebraElem)
  A = parent(x)
  R = base_algebra(A)
  d = base_dimension(A)
  k = truncation_level(A)
  lynd = generate_lyndon_words(k, Vector((1:d)))
  #res = ideal(R,zero(R))
  elem_tts = tensor_sequence(x)
  return ideal(R,[elem_tts[length(w)+1][w...] for w in lynd])
end



function common_ring(R_tensor, R_matrix)
    if R_matrix isa MPolyRing
        return R_matrix
    else
        return R_tensor
    end
end


# Help function for permutations
function permutations_ntuple(v::NTuple{N,Int}) where N
    if N == 1
        return [v]
    end
    out = NTuple{N,Int}[]
    for i in 1:N
        x = v[i]
        rest = ntuple(k -> v[k < i ? k : k+1], N-1)
        for p in permutations_ntuple(rest)
            push!(out, (x, p...))
        end
    end
    return out
end

permutations_1_to_j(j::Int) =
    permutations_ntuple(ntuple(identity, j))


function sig_segment_TA(T::TruncatedTensorAlgebra{R}, v::Vector{E}) where {R,E}

    k = truncation_level(T)
    d = base_dimension(T)

    @assert length(v) == d "dimensions do not match"


    @assert typeof(v[1]) <: E "element types do not match"
    

    T1 = TruncatedTensorAlgebra(base_algebra(T), 1, k, :iis)


    C = sig_mono_TA(T1)

    v_mat = reshape(v, d, 1)


    return matrix_tensorAlg_congruence_TA(v_mat, C)
end


function sig_segment_standard_direction_TA(T::TruncatedTensorAlgebra{R}, _i::Int) where R
    d = base_dimension(T)
    A = base_algebra(T)

    v = one_hot(_i, d, A)
    return sig_segment_TA(T, v)
end




function leading_coefficient_and_zero_TA(p)
    if is_zero(p)
        return leading_coefficient(one(parent(p))) - leading_coefficient(one(parent(p))) # TODO: its zero 
    else
        return leading_coefficient(p)
    end
end



function concatenate_tensors_TA(t1::TruncatedTensorAlgebraElem, t2::TruncatedTensorAlgebraElem)

    @assert parent(t1) === parent(t2) "Parents must match"


    elem1 = t1.elem
    elem2 = t2.elem 

    @assert length(elem1) == length(elem2) "Truncation levels must match"

    new_elem = Vector{Any}(undef, length(elem1))

    for k in 1:length(elem1)
        tensor1 = elem1[k]
        tensor2 = elem2[k]

        new_elem[k] = concatenate_tensors(tensor1, tensor2)
    end

    return TruncatedTensorAlgebraElem(parent(t1), new_elem)
end


function evaluate_to_tensor_TA(
    p::FreeTruncSigAlgMultivElem{T},
    _bs::Vector{TruncatedTensorAlgebraElem{R,E}},
    index::Int
) where {T,R,E}

    bs = deepcopy(_bs)

    # Level 0
    if index == 0
        res = tensor_sequence(bs[1])[1]  
        res[] = leading_coefficient_and_zero_TA(
                  FreeTruncSigAlgMultivElem_to_polynomial(graded_component(p, 0))
               ) * one(res[])
        return res
    end

    num_vars = length(bs)
    A = parent(bs[1])
    k = truncation_level(A)


    fi = FreeTruncSigAlgMultivElem_to_polynomial(graded_component(p, index))


    seq_tensor_seq = [tensor_sequence(b)[2:k+1] for b in bs]


    res = zero(seq_tensor_seq[1][index])


    for t in 1:length(fi)
        fit = collect(terms(fi))[t]

        first_exp  = fit.exps[1][1]
        sample_var = div(first_exp-1, k) + 1
        tensor_var = mod(first_exp-1, k) + 1

        temp = leading_coefficient_and_zero_TA(fit) .* seq_tensor_seq[sample_var][tensor_var]

        for fak in 2:length(fit.exps[1])
            temp_exp   = fit.exps[1][fak]
            sample_var = div(temp_exp-1, k) + 1
            tensor_var = mod(temp_exp-1, k) + 1
            temp = concatenate_tensors_TA(temp, seq_tensor_seq[sample_var][tensor_var])
        end

        res += temp
    end

    return res
end



function evaluate_TA(
    p::FreeTruncSigAlgMultivElem{T},
    bs::Vector{TruncatedTensorAlgebraElem{R,E}}
) where {T,R,E}

    A = parent(bs[1])                     
    k = truncation_level(A)             
    
    res_seq = [evaluate_to_tensor_TA(p, bs, i) for i in 0:k]

    return TruncatedTensorAlgebraElem(parent(bs[1]), res_seq)
end

function combinations_with_replacement(iter, k)
    arr = collect(iter)
    n = length(arr)

    if k == 0
        return [[]]
    end

    res = Vector{Vector{Int}}()
    comb = fill(0, k)

    function backtrack(pos, start)
        if pos > k
            push!(res, copy(comb))
            return
        end
        for i in start:n
            comb[pos] = arr[i]
            backtrack(pos + 1, i)
        end
    end

    backtrack(1, 1)
    return res
end



function sigAxis_TA_ClosedForm(T::TruncatedTensorAlgebra{R}) where R
    if T.sequence_type != :iis
        error("sigAxis_TA only defined for sequence_type = :iis")
    end

    d = base_dimension(T)
    k = truncation_level(T)
    A = base_algebra(T)

    E = typeof(one(A))                
    elem_out = Vector{Array{E}}(undef, k+1)

    elem_out[1] = fill(one(A), ())   

    elem_out[2] = fill(one(A), d)


    for j in 2:k
        dims = ntuple(_ -> d, j)

        tensor_j = fill(zero(A), dims...)

        for idx in combinations_with_replacement(1:d, j)
            counts = Dict{Int,Int}()

            for i in idx
                counts[i] = get(counts, i, 0) + 1
            end

            denom = prod(factorial(c) for c in values(counts))
            value = 1 // denom

            tensor_j[idx...] = try
                A(value)
            catch
                convert(E, value)
            end
        end

        elem_out[j+1] = tensor_j
    end

    return TruncatedTensorAlgebraElem(T, elem_out)
end



function sig_axis_TA(T::TruncatedTensorAlgebra{R}) where R
    k = truncation_level(T)
    d = base_dimension(T) 

    sample = [sig_segment_standard_direction_TA(T, i) for i in 1:d]

    F, s = free_trunc_sig_alg_multiv(k, d)

    chen = prod([free_sig_from_sample(i, F) for i in 1:d])

    return evaluate_TA(chen, sample)
end


function sig_poly_TA(T::TruncatedTensorAlgebra{R}, coeffs::AbstractMatrix{E}) where {R,E}
    d = base_dimension(T)
    @assert size(coeffs,1) == d "Dimensions mismatch"
    k=truncation_level(T)
    A = base_algebra(T)
    m=size(coeffs, 2)
    Tm = TruncatedTensorAlgebra(A,m,k)

    mono_path = sig_mono_TA(Tm)


    poly_path = matrix_tensorAlg_congruence_TA(coeffs, mono_path)

    return poly_path
end

function sig_poly_TA_ARS(T::TruncatedTensorAlgebra{R}, coeffs::AbstractMatrix{E}) where {R,E}
    d = base_dimension(T)
    @assert size(coeffs,1) == d "Dimensions mismatch"
    k=truncation_level(T)
    
    R_tensor = base_algebra(T)
    R_matrix = parent(coeffs[1,1])
    Rnew = common_ring(R_tensor, R_matrix)
    
    Tnew = TruncatedTensorAlgebra(Rnew,d,k)   

    poly_path=polynomial_path_iis_signature(coeffs, k; R=Rnew)

    return TruncatedTensorAlgebraElem(Tnew, poly_path)
end


function sig_pwln_TA_Congruence(T::TruncatedTensorAlgebra{R}, coeffs::AbstractMatrix{E}) where {R,E}
    d = base_dimension(T)
    @assert size(coeffs,1) == d "Dimensions mismatch"
    k=truncation_level(T)
    A = base_algebra(T)
    m=size(coeffs, 2)
    Tm = TruncatedTensorAlgebra(A,m,k)
    return matrix_tensorAlg_congruence_TA(coeffs, sig(Tm,:axis))
end


function sig_pwln_TA_chen(T::TruncatedTensorAlgebra{R}, P::AbstractMatrix{E}) where {R,E}
    d = base_dimension(T)
    @assert size(P,1) == d "Dimensions mismatch"
    m=size(P,2)
    #Old code. dont delete until we create and tested another option for sig_pwln_TA.
    #k1=truncation_level(T)
    #R1 = base_algebra(T)
    #d1=size(P,2)
    #T1 = TruncatedTensorAlgebra(R1,d1,k1)
    #seg_vecs = [P[i+1, :] .- P[i, :] for i in 1:size(P,1)-1]
    #seg_sigs = [sig_segment_TA(T1, seg_vecs[i]) for i in 1:length(seg_vecs)]
    seg_sigs = [sig_segment_TA(T, P[:,i]) for i in 1:m]
    return prod(seg_sigs)
end

function matrix_tensorAlg_congruence_TA(
    v::AbstractVector,
    b::TruncatedTensorAlgebraElem{R,E}
) where {R,E}
    M = reshape(v, 1, length(v))
    return matrix_tensorAlg_congruence_TA(M, b)
end


function matrix_tensorAlg_congruence_TA(
    matrix::AbstractMatrix,
    b::TruncatedTensorAlgebraElem{R,E}
) where {R,E}
    T = parent(b)
    k = truncation_level(T)
    m = base_dimension(T)
    @assert m == ncols(matrix)
    d = nrows(matrix)
    R_tensor = base_algebra(T)
    R_matrix = parent(matrix[1,1])
    Rnew = common_ring(R_tensor, R_matrix)
    Tnew = TruncatedTensorAlgebra(Rnew, d, k, :iis)
    tensorSeq = tensor_sequence(b)
    tensorSeq = [map(x -> Rnew(x), t) for t in tensorSeq]
    resSeq = [
        matrix_tensor_congruence_TA(map(x -> Rnew(x), matrix), t)
        for t in tensorSeq
    ]
    return TruncatedTensorAlgebraElem(Tnew, resSeq)
end


# -------------------------------
# Dispatch Axis sig of membrane
# -------------------------------
function sigAxis_ClosedForm(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    if m * n != base_dimension(T)
        error("m * n != d")
    end

    if T.sequence_type == :p2id
        return sigAxis_p2id_ClosedForm(T, m, n)
    elseif T.sequence_type == :p2
        return sigAxis_p2_ClosedForm(T, m, n)
    else
        error("sequence_type must be :p2id or :p2")
    end
end

function sigAxis_p2id_ClosedForm(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    k = truncation_level(T)
    d=base_dimension(T)
    if m * n != d
        error("m * n != d")
    end

    Tm = TruncatedTensorAlgebra(base_algebra(T), m, k, :iis)
    Tn = TruncatedTensorAlgebra(base_algebra(T), n, k, :iis)

    sig_m = sigAxis_TA_ClosedForm(Tm)
    sig_n = sigAxis_TA_ClosedForm(Tn)
    
    elem_out = Vector{Array{eltype(sig_m.elem[2])}}(undef, k+1)

    elem_out[1] = sig_m.elem[1]

    for j in 1:k
        sigma_m = sig_m.elem[j+1]
        sigma_n = sig_n.elem[j+1]

        tensor_j = Array{eltype(sigma_m)}(undef, ntuple(_ -> m*n, j)...)

        for idx in CartesianIndices(tensor_j)
            linear_idx = Tuple(idx)

            idx_m = ntuple(t -> (div(linear_idx[t]-1, n) % m) + 1, j)
            idx_n = ntuple(t -> (mod(linear_idx[t]-1, n) + 1), j)

            tensor_j[idx] = sigma_m[idx_m...] * sigma_n[idx_n...]
        end

        elem_out[j+1] = tensor_j
    end

    return TruncatedTensorAlgebraElem{R, eltype(sig_m.elem[2])}(T, elem_out)
end


function sigAxis_p2_ClosedForm(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    k = truncation_level(T)
    d=base_dimension(T)

    if m * n != d
        error("m * n != d")
    end


    Tm = TruncatedTensorAlgebra(base_algebra(T), m, k, :iis)
    Tn = TruncatedTensorAlgebra(base_algebra(T), n, k, :iis)

    sig_m = sigAxis_TA_ClosedForm(Tm)
    sig_n = sigAxis_TA_ClosedForm(Tn)

    elem_out = Vector{Any}(undef, k+1)


    elem_out[1] = sig_m.elem[1]

    for j in 1:(k)
        sigma_m = sig_m.elem[j+1]
        sigma_n = sig_n.elem[j+1]

        perms = permutations_1_to_j(j)

        tensor_j = Array{eltype(sigma_m)}(
            undef,
            (ntuple(_ -> m*n, j)..., factorial(j))
        )

        total = (m*n)^j
        perm_idx = 1

        for perm in perms
            for lin in 1:total

                linear_idx = ntuple(
                    t -> ((lin - 1) ÷ (m*n)^(t-1)) % (m*n) + 1,
                    j
                )

                idx_m = ntuple(
                    t -> (div(linear_idx[t] - 1, n) % m) + 1,
                    j
                )

                idx_n = ntuple(
                    t -> (mod(linear_idx[t] - 1, n) + 1),
                    j
                )

                idx_n_perm = ntuple(t -> idx_n[perm[t]], j)

                tensor_j[linear_idx..., perm_idx] =
                    sigma_m[idx_m...] * sigma_n[idx_n_perm...]
            end
            perm_idx += 1
        end

        elem_out[j+1] = tensor_j
    end
    
    return TruncatedTensorAlgebraElem{R, eltype(sig_m.elem[2])}(T, elem_out)
end



# -------------------------------
# Dispatch for Axis Chen Membrane
# -------------------------------
function sigAxis_Chen(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    if m * n != base_dimension(T)
        error("m * n != d")
    end

    if T.sequence_type == :p2id
        return sigAxis_p2id_Chen(T, m, n)
    elseif T.sequence_type == :p2
        return sigAxis_p2_Chen(T, m, n)
    else
        error("sequence_type must be :p2id or :p2")
    end
end

function sigAxis_p2id_Chen(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    k = truncation_level(T)
    d=base_dimension(T)
    if m * n != d
        error("m * n != d")
    end

    Tm = TruncatedTensorAlgebra(base_algebra(T), m, k, :iis)
    Tn = TruncatedTensorAlgebra(base_algebra(T), n, k, :iis)

    
    sig_m = sig_axis_TA(Tm)
    sig_n = sig_axis_TA(Tn)

    elem_out = Vector{Array{eltype(sig_m.elem[2])}}(undef, k+1)

    elem_out[1] = sig_m.elem[1]

    for j in 1:k
        sigma_m = sig_m.elem[j+1]
        sigma_n = sig_n.elem[j+1]

        tensor_j = Array{eltype(sigma_m)}(undef, ntuple(_ -> m*n, j)...)

        for idx in CartesianIndices(tensor_j)
            linear_idx = Tuple(idx)

            idx_m = ntuple(t -> (div(linear_idx[t]-1, n) % m) + 1, j)
            idx_n = ntuple(t -> (mod(linear_idx[t]-1, n) + 1), j)

            tensor_j[idx] = sigma_m[idx_m...] * sigma_n[idx_n...]
        end

        elem_out[j+1] = tensor_j
    end

    return TruncatedTensorAlgebraElem{R, eltype(sig_m.elem[2])}(T, elem_out)
end


function sigAxis_p2_Chen(T::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    k = truncation_level(T)
    d=base_dimension(T)

    if m * n != d
        error("m * n != d")
    end


    Tm = TruncatedTensorAlgebra(base_algebra(T), m, k, :iis)
    Tn = TruncatedTensorAlgebra(base_algebra(T), n, k, :iis)

    sig_m = sig_axis_TA(Tm)
    sig_n = sig_axis_TA(Tn)

    elem_out = Vector{Any}(undef, k+1)

    elem_out[1] = sig_m.elem[1]

    for j in 1:(k)
        sigma_m = sig_m.elem[j+1]
        sigma_n = sig_n.elem[j+1]

        perms = permutations_1_to_j(1:j)

        tensor_j = Array{eltype(sigma_m)}(
            undef,
            (ntuple(_ -> m*n, j)..., factorial(j))
        )

        total = (m*n)^j
        perm_idx = 1

        for perm in perms
            for lin in 1:total

                linear_idx = ntuple(
                    t -> ((lin - 1) ÷ (m*n)^(t-1)) % (m*n) + 1,
                    j
                )

                idx_m = ntuple(
                    t -> (div(linear_idx[t] - 1, n) % m) + 1,
                    j
                )

                idx_n = ntuple(
                    t -> (mod(linear_idx[t] - 1, n) + 1),
                    j
                )

                idx_n_perm = ntuple(t -> idx_n[perm[t]], j)

                tensor_j[linear_idx..., perm_idx] =
                    sigma_m[idx_m...] * sigma_n[idx_n_perm...]
            end
            perm_idx += 1
        end

        elem_out[j+1] = tensor_j
    end
    
    return TruncatedTensorAlgebraElem{R, eltype(sig_m.elem[2])}(T, elem_out)
end


# ==============================================================
# Helper moment path
# ==============================================================
"""
    moment_path_level(R::QQMPolyRing, d::Int, j::Int)

Constructs the moment tensor of level `j` for dimension `d` over the ring `R`.
Each multi-index entry is numerator / denominator * one(R), 
with numerator = prod(indices) and denominator = prod(cumsum(indices)).
"""
function moment_path_level(T::TruncatedTensorAlgebra, d::Int, j::Int)
    dims = ntuple(_ -> d, j)
    T2 = Array{typeof(zero(base_algebra(T)))}(undef, dims...)

    for idx in Iterators.product(ntuple(_ -> 1:d, j)...)
        idx_vec = collect(idx)
        numerator = prod(idx_vec)
        denominator = prod(cumsum(idx_vec))
        T2[idx...] = QQ(numerator, denominator) * one(base_algebra(T))
    end

    return T2
end

# ==============================================================
# Moment membrane for :p2id
# ==============================================================
"""
    moment_membrane_p2id(TTA::TruncatedTensorAlgebra, m::Int, n::Int)

Compute the moment tensors for a TruncatedTensorAlgebra of sequence type `:p2id`.
Returns a new TTA object with `elem[j]` filled for all levels up to `truncation_level`.
"""
function moment_membrane_p2id(TTA::TruncatedTensorAlgebra{R}, m::Int, n::Int) where R
    k = truncation_level(TTA)
    d = base_dimension(TTA)
    
    if m * n != d
        error("m * n must equal the ambient dimension of TTA")
    end

    E = typeof(one(base_algebra(TTA)))
    seq = Vector{Array{E}}(undef, k+1)
    seq[1] = fill(one(base_algebra(TTA)), ())    # array 0-dimensional
    
    for j in 1:k
        s_m = moment_path_level(TTA, m, j)
        s_n = moment_path_level(TTA, n, j)

        dims = ntuple(_ -> m*n, j)
        tensor_j = Array{eltype(s_m)}(undef, dims...)  

        for idx in Iterators.product(ntuple(_ -> 1:(m*n), j)...)
            idx_tuple = Tuple(idx)
            idx_m = [(div(i-1, n) % m) + 1 for i in idx_tuple]
            idx_n = [(mod(i-1, n) + 1) for i in idx_tuple]

            tensor_j[idx_tuple...] = getindex(s_m, idx_m...) * getindex(s_n, idx_n...)
        end

        seq[j+1] = tensor_j
    end

    return TruncatedTensorAlgebraElem(TTA, seq)
end


# ==============================================================
# Moment membrane for :p2
# ==============================================================
"""
    moment_membrane_p2(TTA::TruncatedTensorAlgebra, m::Int, n::Int)

Compute the moment tensors for a TruncatedTensorAlgebra of sequence type `:p2`.
Returns a new TTA object with `elem[j]` filled for all levels up to `truncation_level`.
"""
function moment_membrane_p2(TTA::TruncatedTensorAlgebra, m::Int, n::Int)
    R = base_algebra(TTA)
    k = truncation_level(TTA)
    d = base_dimension(TTA)

    if m * n != d
        error("m * n must equal the ambient dimension of TTA")
    end

    TTA_out = TruncatedTensorAlgebra(
        R, d, k, :p2
    )
    TTA_out.elem = Vector{Any}(undef, k)

    for j in 1:k
        sigma_m = moment_path_level(R, m, j)
        sigma_n = moment_path_level(R, n, j)

        perms = collect(permutations_1_to_j(j))
        tensor_j = Array{QQMPolyRingElem}(undef, (ntuple(_ -> m*n, j)..., factorial(j)))

        # Loop over permutations
        for (perm_idx, perm) in enumerate(perms)
            sigma_n_perm = Array{QQMPolyRingElem}(undef, size(sigma_n)...)
            for idx in Iterators.product(ntuple(_ -> 1:n, j)...)
                idx_perm = idx[perm]
                sigma_n_perm[idx...] = sigma_n[idx_perm...]
            end

            for idx in Iterators.product(ntuple(_ -> 1:(m*n), j)...)
                idx_m = [(div(i-1, n) % m) + 1 for i in idx]
                idx_n = [(mod(i-1, n) + 1) for i in idx]
                tensor_j[idx..., perm_idx] = sigma_m[idx_m...] * sigma_n_perm[idx_n...]
            end
        end

        TTA_out.elem[j] = tensor_j
    end

    return TTA_out
end

"""
    applyMatrixToTTA(A::AbstractMatrix, X::TruncatedTensorAlgebra)

Applies a linear transformation `A` to all levels of a truncated tensor algebra `X`
by mode-product along all modes. Handles both `:p2id` and `:p2` signatures.

- `A` : matrix of size (d_new × d), linear transformation
- `X` : input truncated tensor algebra
Returns a **new** TruncatedTensorAlgebra with updated ambient dimension `d_new`.
"""
function applyMatrixToTTA(
    A::AbstractMatrix,
    X::TruncatedTensorAlgebraElem{R,E}
) where {R,E}

    P = parent(X)
    d_old = base_dimension(P)
    k     = truncation_level(P)
    seq_type = sequence_type(P)

    d_new, dA = size(A)
    @assert dA == d_old "size(A,2) must match ambient dimension"

    R_tensor = base_algebra(P)
    R_matrix = parent(A[1,1])
    Rnew     = common_ring(R_tensor, R_matrix)


    Tnew = TruncatedTensorAlgebra(Rnew, d_new, k, seq_type)


    Anew = map(x -> Rnew(x), A)

    seq_old = tensor_sequence(X)
    seq_old = [map(x -> Rnew(x), T) for T in seq_old]

    resSeq = similar(seq_old)


    resSeq[1] = seq_old[1]

    for j in 2:(k+1)

        T = seq_old[j]
        order = j - 1

        if seq_type == :iis || seq_type == :p2id

            resSeq[j] = matrix_tensor_congruence_TA(Anew, T)

        elseif seq_type == :p2
            dims = ntuple(_ -> d_new, order)
            Tperm = Array{eltype(T)}(undef, (dims..., factorial(order)))

            for perm_idx in 1:factorial(order)
                # Extraer slice puro de shape (d_old,...,d_old)
                T_slice = T[ntuple(_ -> :, order)..., perm_idx]

                # Aplicar congruencia solo sobre las dimensiones tensoriales
                Snew = matrix_tensor_congruence_TA(Anew, T_slice)

                # Guardar en el slot correspondiente
                Tperm[ntuple(_ -> :, order)..., perm_idx] = Snew
            end
            resSeq[j] = Tperm

        else
            error("sequence_type must be :iis, :p2id or :p2")
        end
    end

    return TruncatedTensorAlgebraElem(Tnew, resSeq)
end


"""
    sig2parPoly(T::TruncatedTensorAlgebra, a::AbstractMatrix)

Given a truncated tensor algebra `T` (seq_type `:p2id` or `:p2`), and a matrix of coefficients `a`,
constructs the **moment membrane tensor** in the algebra of `T`, and then applies the matrix `a`
via matrix-tensor congruence. Returns a **new** truncated tensor algebra with updated ambient dimension.

- `T` : input truncated tensor algebra
- `a` : coefficient matrix (d_new × d)
"""
function tensor_to_matrix(A::AbstractArray{T,3}) where {T}
    d, m, n = size(A)

    M = fill(zero(T), d, m*n)

    for k in 1:d, i in 1:m, j in 1:n
        col = (i - 1) * n + j
        M[k, col] = A[k, i, j]
    end

    return M
end

function sig2parPoly_fromMatrix(
    T::TruncatedTensorAlgebra,
    A_tilde::AbstractMatrix{S},
    m::Int,
    n::Int
) where {S}

    d, mn = size(A_tilde)

    d == base_dimension(T) ||
        error("size(A_tilde,1) must equal T.base_dimension")

    mn == m * n ||
        error("size(A_tilde,2) must equal m*n")


    Ttemp = TruncatedTensorAlgebra(
        base_algebra(T),
        m*n,
        truncation_level(T),
        :p2id
    )

    T_moment = moment_membrane_p2id(Ttemp, m, n)

    T_new = applyMatrixToTTA(A_tilde, T_moment)

    return T_new
end

function sig2parPoly_fromMatrix_p2(
    T::TruncatedTensorAlgebra,
    A_tilde::AbstractMatrix{S},
    m::Int,
    n::Int
) where {S}

    d, mn = size(A_tilde)

    d == base_dimension(T) ||
        error("size(A_tilde,1) must equal T.base_dimension")

    mn == m * n ||
        error("size(A_tilde,2) must equal m*n")


    Ttemp = TruncatedTensorAlgebra(
        base_algebra(T),
        m*n,
        truncation_level(T),
        :p2
    )

    T_moment = moment_membrane_p2(Ttemp, m, n)

    T_new = applyMatrixToTTA(A_tilde, T_moment)

    return T_new
end

"""
    sig2parPoly(T::TruncatedTensorAlgebra, A::AbstractArray{S,3})

Apply tensor–matrix congruence to the moment membrane associated with T,
using a rank-3 tensor A ∈ ℝ^{d×m×n} interpreted as a linear operator

    ℝ^m ⊗ ℝ^n → ℝ^d.

Returns:
- T_new :: TruncatedTensorAlgebra
- (m, n) :: dimensions of the moment membrane
"""
function sig2parPoly(
    T::TruncatedTensorAlgebra,
    A::AbstractArray{S,3}
) where {S}


    d, m, n = size(A)

    d == base_dimension(T) ||
       error("size(A,1) must equal T.base_dimension")
    
    Ttemp=TruncatedTensorAlgebra(base_algebra(T), m*n, truncation_level(T), :p2id)

    T_moment = moment_membrane_p2id(Ttemp, m, n)

    A_tilde = fill(zero(parent(A[1,1,1])), d, m * n)


    @inbounds for kidx in 1:d
        for i in 1:m
            for j in 1:n
                col = (i - 1) * n + j
                A_tilde[kidx, col] = A[kidx, i, j]
            end
        end
    end


    T_new = applyMatrixToTTA(A_tilde, T_moment)

    return T_new
end


function sig2parPoly_p2(
    T::TruncatedTensorAlgebra,
    A::AbstractArray{S,3}
) where {S}
    d, m, n = size(A)
    d == base_dimension(T) ||
        error("size(A,1) must equal T.base_dimension")
    Ttemp = TruncatedTensorAlgebra(base_algebra(T), m*n, truncation_level(T), :p2)
    T_moment = moment_membrane_p2(Ttemp, m, n)
    A_tilde = fill(zero(parent(A[1,1,1])), d, m * n)
    @inbounds for kidx in 1:d
        for i in 1:m
            for j in 1:n
                col = (i - 1) * n + j
                A_tilde[kidx, col] = A[kidx, i, j]
            end
        end
    end
    T_new = applyMatrixToTTA(A_tilde, T_moment)
    return T_new
end


function sig_pwbln_p2id_Congruence_fromTensor(
    T::TruncatedTensorAlgebra,
    membrane_tensor::AbstractArray{S,3},
    m::Int,
    n::Int
) where {S}

    size(membrane_tensor,1) == m || error("First dimension must be m")
    size(membrane_tensor,2) == n || error("Second dimension must be n")

    d_new = size(membrane_tensor,3)


    coef2 = Matrix{S}(undef, d_new, m*n)

    for di in 1:d_new
        cont = 0
        for i in 1:m
            for j in 1:n
                cont += 1
                coef2[di, cont] = membrane_tensor[i, j, di]
            end
        end
    end

    A = coef2

    d_old = m*n
    k_old = truncation_level(T)
    R_old = base_algebra(T)

    T_old = TruncatedTensorAlgebra(R_old, d_old, k_old, :p2)

    T_axis = sigAxis_p2_ClosedForm(T_old, m, n)

    T_new = applyMatrixToTTA(A, T_axis)

    return T_new
end

function sig_pwbln_p2_Congruence_fromTensor(
    T::TruncatedTensorAlgebra,
    membrane_tensor::AbstractArray{S,3},
    m::Int,
    n::Int
) where {S}

    size(membrane_tensor,1) == m || error("First dimension must be m")
    size(membrane_tensor,2) == n || error("Second dimension must be n")

    d_new = size(membrane_tensor,3)


    coef2 = Matrix{S}(undef, d_new, m*n)

    for di in 1:d_new
        cont = 0
        for i in 1:m
            for j in 1:n
                cont += 1
                coef2[di, cont] = membrane_tensor[i, j, di]
            end
        end
    end

    A = coef2

    d_old = m*n
    k_old = truncation_level(T)
    R_old = base_algebra(T)

    T_old = TruncatedTensorAlgebra(R_old, d_old, k_old, :p2)

    T_axis = sigAxis_p2_ClosedForm(T_old, m, n)

    T_new = applyMatrixToTTA(A, T_axis)

    return T_new
end



function sig_pwbln_p2id_Congruence(
    T::TruncatedTensorAlgebra,
    A::AbstractMatrix{S},  
    m::Int, n::Int         # dimension
) where {S}
    
    size(A,2) == (m*n) || error("A must have size (d_new, m*n")
    
    d_old=m*n
    k_old=truncation_level(T)
    R_old=base_algebra(T)
    T_old=TruncatedTensorAlgebra(R_old, d_old, k_old, :p2id)

    T_axis = sigAxis_p2id_ClosedForm(T_old, m, n)
    T_new = applyMatrixToTTA(A, T_axis)

    return T_new
end

function sig_pwbln_p2_Congruence(
    T::TruncatedTensorAlgebra,
    A::AbstractMatrix{S},  
    m::Int, n::Int         # dimension
) where {S}
    
    size(A,2) == (m*n) || error("A must have size (d_new, m*n")
    
    d_old=m*n
    k_old=truncation_level(T)
    R_old=base_algebra(T)
    T_old=TruncatedTensorAlgebra(R_old, d_old, k_old, :p2)

    T_axis = sigAxis_p2_ClosedForm(T_old, m, n)
    T_new = applyMatrixToTTA(A, T_axis)

    return T_new
end


function row_sum(matrix)
    return sum(matrix, dims=2) 
end


function column_sum(matrix)
    return sum(matrix, dims=1) 
end

function weighted_shift(matrix::AbstractMatrix)
    m, n = size(matrix)

    out = fill(zero(eltype(matrix)), m, n)
    for i in 2:m
        for j in 2:n
            coef = 1 // ((i-1) * (j-1))
            out[i, j] = coef * matrix[i-1, j-1]
        end
    end
    return out
end



function sig_lott(matrixSeq::Vector{<:AbstractMatrix}, word::Vector{Int})
    d = length(matrixSeq)        
    m, n = size(matrixSeq[1])    
    k = length(word)                   

    T = eltype(matrixSeq[1])
    
    sheet = fill(zero(T), m, n, k+1, k+1)
    for i in 1:m
        for j in 1:n
            sheet[i, j, 1, 1] += one(T)
        end
    end

    for p in 1:k
        letter = word[p]
        mat = matrixSeq[letter]

        for i in 1:m
            for j in 1:n
                sheet[i,j,:,:] .*= mat[i,j]
                sheet[i,j,:,:] = weighted_shift(sheet[i,j,:,:])
            end
        end

        for i in 1:(m-1)
            rsum = [vec(row_sum(sheet[i,j,:,:])) for j in 1:n]
            for j in 1:n
                sheet[i+1,j,:,1] .+= rsum[j]
            end
        end

        for j in 1:(n-1)
            csum = [vec(column_sum(sheet[i,j,:,:])) for i in 1:m]
            for i in 1:m
                sheet[i,j+1,1,:] .+= csum[i]
            end
        end
    end

    out = sum(row_sum(column_sum(sheet[m,n,:,:])))
    return out
end


function sigPiecewiseBilinear_TA(
    T::TruncatedTensorAlgebra{R},
    coef2::AbstractMatrix,
    shape
) where R

    if T.sequence_type != :p2id
        error("sigPiecewiseBilinear_TA only defined for sequence_type = :p2id")
    end
    if isa(shape, AbstractVector) && length(shape) == 2
        m, n = shape[1], shape[2]
    elseif isa(shape, Tuple) && length(shape) == 2
        m, n = shape
    else
        error("shape must be a Tuple or Vector of length 2")
    end
    d = size(coef2, 1)

    if size(coef2, 2) != (m*n)
        error("size(coef,2) does not match number of membranes")
    end

    k = truncation_level(T)
    Rbase = base_algebra(T)
    R_matrix = parent(coef2[1,1])


    Rnew = common_ring(Rbase, R_matrix)

    E = typeof(one(Rnew))

    membrane = Vector{Matrix{E}}(undef, d)
    for di in 1:d
        M = Matrix{E}(undef, m, n)
        cont = 0
        for i in 1:m
            for j in 1:n
                cont += 1
                M[i,j] = coef2[di, cont]
            end
        end
        membrane[di] = M
    end

    elem_out = Vector{Array{E}}(undef, k+1)

    elem_out[1] = fill(one(Rnew), ())

    function compute_level(r)
        tensor_dims = ntuple(_ -> d, r)

        Tlevel = fill(zero(E), tensor_dims...)


        function loop_word(current::Vector{Int}, level::Int)
            if level > r
                val = sig_lott(membrane, copy(current))

                Tlevel[Tuple(current)...] = try
                    convert(E, val)
                catch
                    val
                end
            else
                for letter in 1:d
                    current[level] = letter
                    loop_word(current, level + 1)
                end
            end
        end

        loop_word(fill(0, r), 1)
        return Tlevel
    end

    for j in 1:k
        elem_out[j+1] = compute_level(j)
    end
    T2=TruncatedTensorAlgebra(Rnew, d, k, :p2id)
    return TruncatedTensorAlgebraElem(T2, elem_out)
end

function sigPiecewiseBilinear_TA_p2(
    T::TruncatedTensorAlgebra{R},
    coef2::AbstractMatrix,
    shape
) where R
    if T.sequence_type != :p2
        error("sigPiecewiseBilinear_TA_p2 only defined for sequence_type = :p2")
    end
    if isa(shape, AbstractVector) && length(shape) == 2
        m, n = shape[1], shape[2]
    elseif isa(shape, Tuple) && length(shape) == 2
        m, n = shape
    else
        error("shape must be a Tuple or Vector of length 2")
    end
    d = size(coef2, 1)
    if size(coef2, 2) != (m * n)
        error("size(coef,2) does not match number of membranes")
    end
    k      = truncation_level(T)
    Rbase  = base_algebra(T)
    R_matrix = parent(coef2[1, 1])
    Rnew   = common_ring(Rbase, R_matrix)
    E      = typeof(one(Rnew))

    # build membrane exactly as in p2id version
    membrane = Vector{Matrix{E}}(undef, d)
    for di in 1:d
        M    = Matrix{E}(undef, m, n)
        cont = 0
        for i in 1:m
            for j in 1:n
                cont += 1
                M[i, j] = coef2[di, cont]
            end
        end
        membrane[di] = M
    end

    elem_out    = Vector{Array{E}}(undef, k + 1)
    elem_out[1] = fill(one(Rnew), ())

    function compute_level_p2(r)
        perms    = collect(permutations_1_to_j(r))   # all π ∈ Σ_r, length = r!
        n_perms  = length(perms)
        # shape: (d, d, ..., d, r!) — r indices of size d + 1 index of size r!
        tensor_dims = (ntuple(_ -> d, r)..., n_perms)
        Tlevel      = fill(zero(E), tensor_dims...)

        function loop_word(current::Vector{Int}, level::Int)
            if level > r
                for (pidx, perm) in enumerate(perms)
                    # permute the word according to π: apply π to the time-axis
                    permuted_word = [current[perm[q]] for q in 1:r]
                    val = sig_lott(membrane, permuted_word)
                    idx = (Tuple(current)..., pidx)
                    Tlevel[idx...] = try
                        convert(E, val)
                    catch
                        val
                    end
                end
            else
                for letter in 1:d
                    current[level] = letter
                    loop_word(current, level + 1)
                end
            end
        end

        loop_word(fill(0, r), 1)
        return Tlevel
    end

    for j in 1:k
        elem_out[j + 1] = compute_level_p2(j)
    end

    T2 = TruncatedTensorAlgebra(Rnew, d, k, :p2)
    return TruncatedTensorAlgebraElem(T2, elem_out)
end

function sigPiecewiseBilinear_fromTensor_TA(
    T::TruncatedTensorAlgebra{R},
    membrane_tensor::AbstractArray{S,3},
    shape
) where {R,S}

    if T.sequence_type != :p2id
        error("sigPiecewiseBilinear_TA only defined for sequence_type = :p2id")
    end

    if isa(shape, AbstractVector) && length(shape) == 2
        m, n = shape[1], shape[2]
    elseif isa(shape, Tuple) && length(shape) == 2
        m, n = shape
    else
        error("shape must be a Tuple or Vector of length 2")
    end

    size(membrane_tensor,1) == m || error("First dimension must be m")
    size(membrane_tensor,2) == n || error("Second dimension must be n")

    d = size(membrane_tensor,3)

    k = truncation_level(T)
    Rbase = base_algebra(T)
    R_tensor = parent(membrane_tensor[1,1,1])

    Rnew = common_ring(Rbase, R_tensor)
    E = typeof(one(Rnew))

    membrane = Vector{Matrix{E}}(undef, d)

    for di in 1:d
        M = Matrix{E}(undef, m, n)
        for i in 1:m
            for j in 1:n
                M[i,j] = convert(E, membrane_tensor[i,j,di])
            end
        end
        membrane[di] = M
    end

    elem_out = Vector{Array{E}}(undef, k+1)

    elem_out[1] = fill(one(Rnew), ())

    function compute_level(r)
        tensor_dims = ntuple(_ -> d, r)
        Tlevel = fill(zero(E), tensor_dims...)

        function loop_word(current::Vector{Int}, level::Int)
            if level > r
                val = sig_lott(membrane, copy(current))
                Tlevel[Tuple(current)...] = try
                    convert(E, val)
                catch
                    val
                end
            else
                for letter in 1:d
                    current[level] = letter
                    loop_word(current, level + 1)
                end
            end
        end

        loop_word(fill(0, r), 1)
        return Tlevel
    end

    for j in 1:k
        elem_out[j+1] = compute_level(j)
    end

    T2 = TruncatedTensorAlgebra(Rnew, d, k, :p2id)
    return TruncatedTensorAlgebraElem(T2, elem_out)
end

function sigPiecewiseBilinear_fromTensor_TA_p2(
    T::TruncatedTensorAlgebra{R},
    membrane_tensor::AbstractArray{S,3},
    shape
) where {R,S}
    if T.sequence_type != :p2
        error("sigPiecewiseBilinear_fromTensor_TA_p2 only defined for sequence_type = :p2")
    end
    if isa(shape, AbstractVector) && length(shape) == 2
        m, n = shape[1], shape[2]
    elseif isa(shape, Tuple) && length(shape) == 2
        m, n = shape
    else
        error("shape must be a Tuple or Vector of length 2")
    end
    size(membrane_tensor, 1) == m || error("First dimension must be m")
    size(membrane_tensor, 2) == n || error("Second dimension must be n")
    d      = size(membrane_tensor, 3)
    k      = truncation_level(T)
    Rbase  = base_algebra(T)
    R_tensor = parent(membrane_tensor[1, 1, 1])
    Rnew   = common_ring(Rbase, R_tensor)
    E      = typeof(one(Rnew))

    membrane = Vector{Matrix{E}}(undef, d)
    for di in 1:d
        M = Matrix{E}(undef, m, n)
        for i in 1:m
            for j in 1:n
                M[i, j] = convert(E, membrane_tensor[i, j, di])
            end
        end
        membrane[di] = M
    end

    elem_out    = Vector{Array{E}}(undef, k + 1)
    elem_out[1] = fill(one(Rnew), ())

    function compute_level_p2(r)
        perms       = collect(permutations_1_to_j(r))
        n_perms     = length(perms)
        tensor_dims = (ntuple(_ -> d, r)..., n_perms)
        Tlevel      = fill(zero(E), tensor_dims...)

        function loop_word(current::Vector{Int}, level::Int)
            if level > r
                for (pidx, perm) in enumerate(perms)
                    permuted_word = [current[perm[q]] for q in 1:r]
                    val = sig_lott(membrane, permuted_word)
                    idx = (Tuple(current)..., pidx)
                    Tlevel[idx...] = try
                        convert(E, val)
                    catch
                        val
                    end
                end
            else
                for letter in 1:d
                    current[level] = letter
                    loop_word(current, level + 1)
                end
            end
        end

        loop_word(fill(0, r), 1)
        return Tlevel
    end

    for j in 1:k
        elem_out[j + 1] = compute_level_p2(j)
    end

    T2 = TruncatedTensorAlgebra(Rnew, d, k, :p2)
    return TruncatedTensorAlgebraElem(T2, elem_out)
end


function weighted_shift_1d(pvec::AbstractVector)
    n = length(pvec)
    out = fill(zero(eltype(pvec)), n)
    for i in 2:n
        coef = 1 // (i - 1)
        out[i] = coef * pvec[i-1]
    end
    return out
end


function sig_lott_1d(pathSeq::Vector{<:AbstractVector}, word::Vector{Int})
    d = length(pathSeq)
    n = length(pathSeq[1])
    k = length(word)
    T = eltype(pathSeq[1])

    sheet = fill(zero(T), n, k+1)
    for j in 1:n
        sheet[j, 1] = one(T)
    end

    for p in 1:k
        letter = word[p]
        pvec = pathSeq[letter]

        for j in 1:n
            sheet[j, :] .*= pvec[j]
            sheet[j, :] = weighted_shift_1d(sheet[j, :])
        end

        for j in 1:(n-1)
            sheet[j+1, 1] += sum(sheet[j, :])  
        end
    end

    return sum(sheet[n, :])
end


function sig_pwln_TA_LS26(
    T::TruncatedTensorAlgebra{R},
    coef1::AbstractMatrix   
) where {R}

    if T.sequence_type != :iis
        error("sig_pwln_TA_LS only defined for sequence_type = :iis")
    end

    d, n = size(coef1)

    if d != base_dimension(T)
        error("coef1 rows must match the algebra dimension d")
    end

    k  = truncation_level(T)
    Rbase    = base_algebra(T)
    R_matrix = parent(coef1[1, 1])
    Rnew     = common_ring(Rbase, R_matrix)
    E        = typeof(one(Rnew))

    path = Vector{Vector{E}}(undef, d)
    for di in 1:d
        path[di] = [coef1[di, i] for i in 1:n]
    end

    elem_out    = Vector{Array{E}}(undef, k+1)
    elem_out[1] = fill(one(Rnew), ())   

    function compute_level(r)
        tensor_dims = ntuple(_ -> d, r)
        Tlevel = fill(zero(E), tensor_dims...)

        function loop_word(current::Vector{Int}, level::Int)
            if level > r
                val = sig_lott_1d(path, copy(current))
                Tlevel[Tuple(current)...] = try
                    convert(E, val)
                catch
                    val
                end
            else
                for letter in 1:d
                    current[level] = letter
                    loop_word(current, level + 1)
                end
            end
        end

        loop_word(fill(0, r), 1)
        return Tlevel
    end

    for j in 1:k
        elem_out[j+1] = compute_level(j)
    end

    T2 = TruncatedTensorAlgebra(Rnew, d, k, :iis)
    return TruncatedTensorAlgebraElem(T2, elem_out)
end
