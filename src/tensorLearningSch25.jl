export
#  tensor_learning_3,
  recover

#function generic_transform(_d,_max_int=10)
#   A = QQ.(rand(-1*_max_int:_max_int, _d, _d))
#   if is_invertible( matrix(QQ,A))
#     return A
#   else
#     return generic_transform(_d,_max_int)
#   end
#end

function toTriangularTrafo_last_2x2_up(G)
  d = size(G)[1]
  Q = Matrix(identity_matrix(QQ,d))
  if G[d,d-1,d-1] == G[d-1,d,d-1] 
    return Q
  end
  if G[d,d-1,d]!=G[d-1,d,d]
    lambda23 = (G[d,d-1,d-1]- G[d-1,d,d-1])*inv(-G[d, d-1, d]+G[d-1, d, d])
    Q1 = identity_matrix(QQ,d) + lambda23.*QQMatrixE(d,d-1,d)
    add_slice_all_3!(G,lambda23,d,d-1)
    return Matrix(Q1)
  else
    swap_slice_all_3!(G,d-1,d)
    Q0 = Matrix(block_diagonal_matrix([identity_matrix(QQ,d-2),permutation_matrix(QQ,symmetric_group(2)[1])]))
    return Q0
  end
  println("PROBLEM s = d-1")
  return Q0
end

function generic_last_3x3(d)
  return Array(block_diagonal_matrix([identity_matrix(QQ,d-3),matrix(QQ,generic_transform_GL(3,2))]))
end

function toTriangularTrafo_last_3x3_up(G)
  d = size(G)[1]
  if G[d, d-1, d]==G[d-1, d, d]
    Qrand = generic_last_3x3(d)
    matrix_tensor_congruence_3!(Qrand,G)
    return Matrix(toTriangularTrafo_last_3x3_up(G))*Matrix(Qrand)
  end
  lambda23 = (G[d,d-1,d-1]- G[d-1,d,d-1])*inv(-G[d, d-1, d]+G[d-1, d, d])
  Q1 = identity_matrix(QQ,d) + lambda23.*QQMatrixE(d,d-1,d)
  Q = Matrix(Q1)
  add_slice_all_3!(G,lambda23,d,d-1)
  lambda13 = (G[d,d-1,d-2]- G[d-1,d,d-2])*inv(-G[d, d-1, d]+G[d-1, d, d])
  Q2 = identity_matrix(QQ,d) + lambda13.*QQMatrixE(d,d-2,d)
  Q = Matrix(Q2)*Q
  add_slice_all_3!(G,lambda13,d,d-2)
  Q3 = identity_matrix(QQ,d)
  if G[d-1, d-2,d-1]==G[d-2, d-1, d-1]
    Qrand = generic_last_3x3(d)
    matrix_tensor_congruence_3!(Qrand,G)
    return Matrix(toTriangularTrafo_last_3x3_up(G))*Matrix(Qrand)*Q
  end
  lambda12 = (G[d-1,d-2,d-2]- G[d-2,d-1,d-2])*inv(-G[d-1, d-2,d-1]+G[d-2, d-1, d-1])
  Q3 = identity_matrix(QQ,d) + lambda12.*QQMatrixE(d,d-2,d-1)
  add_slice_all_3!(G,lambda12,d-1,d-2)
  Q = Matrix(Q3)*Q
  if G[d-2,d-2,d-2]==zero(QQ)
    Qrand = generic_last_3x3(d)
    matrix_tensor_congruence_3!(Qrand,G)
    return Matrix(toTriangularTrafo_last_3x3_up(G))*Matrix(Qrand)*Q
  else
    return Q
  end
end

function linrel_mat_vec(G,s)
  d = size(G)[1]
  nreq = binomial(d-1,2)
  nrun = d - s
  #M = zeros(QQ,nreq,nrun)
  #B = zeros(QQ,nreq)
  M = fill(zero(QQ), nreq, nrun)
  B = fill(zero(QQ), nreq)
  i = 1
  for _a in (s+1:d)
    for _b in (_a+1:d)
      M[i,:] = [(G[_a,_b,s+gamma]-G[_b,_a,s+gamma]) for gamma in (1:nrun)]
      B[i] = G[_b,_a,s] - G[_a,_b,s]
      i = i + 1
    end
  end
  return M,B
end

function toTriangularTrafo_last_sxs_up_RAND(s,G)
  d = size(G)[1]
  T = identity_matrix(QQ,d)
  M,B = linrel_mat_vec(G,s)
  for att in (1:100) # Et hätt noch emmer jot jejange
    issolv,x = can_solve_with_solution(matrix(QQ,M),B,side=:right)
    if issolv && rank(matrix(QQ,M))==d-s
      Qup = Matrix(identity_matrix(QQ,d))
      Qup[s,s+1:d] = x
      for j in (s+1:d)
        add_slice_all_3!(G,Qup[s,j],j,s)
      end
      return Matrix(Qup)*Matrix(T)
    end
    T_temp = Array(block_diagonal_matrix([identity_matrix(QQ,s-1),matrix(QQ,generic_transform_GL(d-s+1,3))]))
    matrix_tensor_congruence_3!(Array(T_temp),G)
    T = Matrix(T_temp)*Matrix(T)
    M,B = linrel_mat_vec(G,s)
  end
  print("ERROR: we tried 100 random coordinate changes and it didn't suffice at s=")
  println(s)
  return identity_matrix(QQ,d)
end

#function QQMatrixE(m::Int,i::Int,j::Int)
#  @assert i<=m
#  @assert j<=m
#  M = zero_matrix(QQ,m,m)
#  M[i,j] = one(QQ)
#  return M
#end

function diagTraf(s::Int,G1)
  d = size(G1)[1]
  P2 = (inv(root(G1[s,s,s],3))-1)*QQMatrixE(d,s,s)
  return P2 + identity_matrix(QQ,d)
end

function lowTraf(s::Int,G2)
  d = size(G2)[1]
  P3 = identity_matrix(QQ,d)
  for j in (s+1:d)
    P3[j,s] = -G2[j,s,s]
  end
  return P3
end

function tensor_learning_3(G_)
  G = copy(G_)
  d = size(G)[1]
  Q = Matrix(identity_matrix(QQ,d))
  for s in (1:d)
    if s == d
       Qup = identity_matrix(QQ,d)
    elseif s == d-2
       Qup = toTriangularTrafo_last_3x3_up(G)
    elseif s == d-1
       Qup = toTriangularTrafo_last_2x2_up(G)
    else
       Qup = toTriangularTrafo_last_sxs_up_RAND(s,G)
    end
    Qdiag = diagTraf(s,G)
    multiply_slice_all_3!(G,inv(root(G[s,s,s],3)),s)
    Qlow = lowTraf(s,G)
    for j in (s+1:d)
      add_slice_all_3!(G,Qlow[j,s],s,j)
    end
    Q = Matrix(Qlow)*Matrix(Qdiag)*Matrix(Qup)*Matrix(Q)
  end
  return Q
end

function leading_coefficient_and_zero_TL(p)
    if is_zero(p)
        return leading_coefficient(one(parent(p))) - leading_coefficient(one(parent(p))) # TODO: its zero 
    else
        return leading_coefficient(p)
    end
end



"""
    recover(S::TruncatedTensorAlgebraElem; 
            C::Union{TruncatedTensorAlgebraElem, Nothing}=nothing,
            algorithm::Symbol = :default)

Recover a linear transformation matrix `A` from a truncated signature `S`, solving
the **tensor learning** (path recovery) inverse problem:
```math
S = A \\cdot C
```

where `C` is a known core tensor (typically the signature of a canonical axis path)
and `A` is the unknown invertible matrix encoding the path.

The uniqueness of this solution (over the reals) is guaranteed by [PSS2019, Theorem 6.2]
under the assumption that `S` lies in the orbit of `C` under matrix-tensor congruence.

See: L. Schmitz, *An Efficient Algorithm for Tensor Learning*, arXiv:2512.14218.

# Arguments
- `S::TruncatedTensorAlgebraElem`: The observed truncated signature to invert.
- `C::Union{TruncatedTensorAlgebraElem, Nothing}`: The core tensor representing the
  signature of the canonical axis path. Required for `:default` and `:Buchberger`.
- `algorithm::Symbol`: Algorithm used to recover `A`. Options:
  - `:default` / `:Buchberger` — solves the system `S = A*C` via Gröbner basis
    computation over `QQ`. Requires `C`, and the resulting ideal must be
    zero-dimensional of degree 1.
  - `:Sch25` — efficient algorithm from arXiv:2512.14218, based on symbolic
    multilinear algebra and Gauß transformations on the third-order signature tensor.
    Requires `base_algebra(parent(S)) == QQ` and truncation level `k = 3`.

# Returns
- For `:default` / `:Buchberger`: a matrix of leading coefficients representing `A`,
  recovered via normal forms modulo the ideal.
- For `:Sch25`: the inverse of the transformation matrix computed by `tensor_learning_3`,
  i.e., `A = inv(tensor_learning_3(6 * S₃))` where `S₃` is the level-3 tensor of `S`.

# Throws
- `AssertionError` if `C` is not provided when using `:default` or `:Buchberger`.
- `AssertionError` if the resulting ideal is not zero-dimensional or not of degree 1
  (when using `:default` / `:Buchberger`).
- `AssertionError` if `base_algebra` is not `QQ` or `k ≠ 3` (when using `:Sch25`).
- `ErrorException` if an unknown algorithm symbol is passed.
"""
function recover(S::TruncatedTensorAlgebraElem; 
                 C::Union{TruncatedTensorAlgebraElem, Nothing}=nothing,
                 algorithm::Symbol = :default)

    d = base_dimension(parent(S))
    k = truncation_level(parent(S))

    if algorithm == :default || algorithm == :Buchberger
        @assert C !== nothing "Argument C must be provided for algorithm=:default"
        m = base_dimension(parent(C))  
      
        R, a = polynomial_ring(QQ, :a => (1:d, 1:m))

        I = ideal(R, vec(S - a*C))

        @assert dim(I) == 0 "Ideal dimension should be 0"
        @assert degree(I) == 1 "Ideal degree should be 1"

   
        return leading_coefficient_and_zero_TL.(normal_form.(a, Ref(I)))

    elseif algorithm == :Sch25
        @assert base_algebra(parent(S)) == QQ "Ground algebra must be QQ for Sch25"
        @assert k == 3 "Algorithm Sch25 relys on truncation level k=3"

        S_array =  6 .* S.elem[4] 
        return inv(tensor_learning_3(S_array))

    else
        error("Unknown algorithm: $algorithm")
    end
end