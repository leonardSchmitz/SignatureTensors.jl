export 
  free_trunc_sig_alg_multiv, 
  free_sig_from_sample, 
  gens_in_shape, 
  free_sig_bary, 
  graded_component, 
  #bary_poly, 
  FreeTruncSigAlgMultivElem_to_polynomial, 
  bary_defining_polynomial_system

##################### 
# typesetting 
##################### 


# for function multivariate_free_signature_algebra(_trunc_level::Int,_num_samples::Int)
function _magic_symbols_sig_multivariate(trunc_level::Int,  
                                         num_samples::Int, 
                                         invert_bary::Bool = false)
  s = Array{String}(undef, trunc_level,num_samples)
  for i in 1:trunc_level
     for j in 1:num_samples
      if i > 9 || j > 9
        #s[i, j] = "s$(_index_number(i))₋$(_index_number(j))"
        s[i, j] = "s$(_index_number(j))₋$(_superscript_number(i))"
      else
        #s[i, j] = "s$(_index_number(i))$(_index_number(j))"
        s[i, j] = "s$(_index_number(j))$(_superscript_number(i))"
      end
    end
  end
  if invert_bary
     for i in 1:trunc_level
        s[i, num_samples] = "y$(_superscript_number(i))"
     end
  end
  return s
end

#############################
# structs
#############################
"""
    FreeTruncSigAlgMultiv{T <: FieldElem}

Represents a free truncated multivariate signature algebra of level `trunc_level` 
with `num_samples` sample elements. Used for barycenter computations and graded components.

# Fields
- `free_alg::FreeAssociativeAlgebra{T}`: underlying free associative algebra
- `trunc_level::Int`: truncation level
- `num_samples::Int`: number of sample elements
- `with_bary::Bool`: whether the algebra includes a barycenter element

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
"""
struct FreeTruncSigAlgMultiv{T <:FieldElem} 
  free_alg::FreeAssociativeAlgebra{T}
  trunc_level::Int
  num_samples::Int
  with_bary::Bool
end

free_algebra(F::FreeTruncSigAlgMultiv) = F.free_alg
truncation_level(F::FreeTruncSigAlgMultiv) = F.trunc_level
number_of_samples(F::FreeTruncSigAlgMultiv) = F.num_samples
with_bary(F::FreeTruncSigAlgMultiv) = F.with_bary


"""
    FreeTruncSigAlgMultiv{T <: FieldElem}

Represents a free truncated multivariate signature algebra of level `trunc_level` 
with `num_samples` sample elements. Used for barycenter computations and graded components.

# Fields
- `free_alg::FreeAssociativeAlgebra{T}`: underlying free associative algebra
- `trunc_level::Int`: truncation level
- `num_samples::Int`: number of sample elements
- `with_bary::Bool`: whether the algebra includes a barycenter element

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
"""
FreeTruncSigAlgMultivElem

struct FreeTruncSigAlgMultivElem{T <: FieldElem}
  parent::FreeTruncSigAlgMultiv{T}
  elem::FreeAssociativeAlgebraElem{T}
end

"""
    free_trunc_sig_alg_multiv(trunc_level::Int, num_samples::Int)
    free_trunc_sig_alg_multiv(trunc_level::Int, num_samples::Int, with_bary::Bool=true)
Constructs a `FreeTruncSigAlgMultiv` algebra and returns a tuple `(F, s)` where
`s` is an array of `FreeTruncSigAlgMultivElem` corresponding to the sample elements.

# Arguments
- `trunc_level::Int`: truncation level
- `num_samples::Int`: number of sample elements
- `with_bary::Bool`: whether to include a barycenter element

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
    s1 = s[1,1]
"""
free_trunc_sig_alg_multiv

function free_trunc_sig_alg_multiv(_trunc_level::Int, 
                                   _num_samples::Int,
                                   _with_bary::Bool = true)
  if _with_bary 
    F, s = free_associative_algebra(QQ, _magic_symbols_sig_multivariate(_trunc_level,_num_samples + 1, _with_bary))
  else 
    F, s = free_associative_algebra(QQ, _magic_symbols_sig_multivariate(_trunc_level,_num_samples, _with_bary))
  end
  TFA = FreeTruncSigAlgMultiv(F, _trunc_level,_num_samples,_with_bary)
  s = map(x -> FreeTruncSigAlgMultivElem(TFA, x), s)
  return TFA, s
end

Base.parent(a::FreeTruncSigAlgMultivElem) = a.parent
FreeTruncSigAlgMultivElem_to_polynomial(a::FreeTruncSigAlgMultivElem) = a.elem

function trunc(f::FreeAssociativeAlgebraElem,_trunc_level::Int)
  #A = parent(f)
  res = 0*f
  if iszero(f)
    return res
  end
  for i in (1:length(f))
    exp_vec = (f.exps)[i].-1
    mod_exp_vec = exp_vec.%_trunc_level
    if sum(mod_exp_vec.+1) <= _trunc_level
      res += collect(terms(f))[i]
    end
  end
  return res 
end

"""
    graded_component(f::FreeTruncSigAlgMultivElem, ell::Int)

Projects the element `f` to its `ell`-th graded component according to the level grading.

# Example
    g = free_sig_bary(F) + free_sig_from_sample(1,F)
    f1 = graded_component(g, 1)
"""
function graded_component(f::FreeAssociativeAlgebraElem,ell::Int,_trunc_level::Int)
  res = 0*f
  if iszero(f)
    return res
  end
  for i in (1:length(f))
    exp_vec = (f.exps)[i].-1
    mod_exp_vec = exp_vec.%_trunc_level
    if sum(mod_exp_vec.+1) == ell
      res += collect(terms(f))[i]
    end
  end
  return res
end

function graded_component(f::FreeTruncSigAlgMultivElem,ell::Int)
  A = parent(f)
  temp = graded_component(f.elem,ell,truncation_level(A))
  return FreeTruncSigAlgMultivElem(A,temp)
end

function Base.show(io::IO, ::MIME"text/plain",f::FreeTruncSigAlgMultivElem)
  A = parent(f) 
  #trunc_level = A.trunc_level
  trunc_level = truncation_level(A)
  #print(io,"$(f.elem) + O($(trunc_level+1))")
  print(io,"$(f.elem)")
end

#############################
# special constructors
#############################

function Base.one(_T::FreeTruncSigAlgMultiv)
  return FreeTruncSigAlgMultivElem(_T,one(_T.free_alg))
end

function Base.zero(_T::FreeTruncSigAlgMultiv)
  return FreeTruncSigAlgMultivElem(_T,zero(_T.free_alg))
end

"""
    free_sig_from_sample(sample_index::Int, F::FreeTruncSigAlgMultiv)

Returns the `FreeTruncSigAlgMultivElem` corresponding to the `sample_index`-th sample
element in the algebra `F`.

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
    s1 = free_sig_from_sample(1, F)
"""
function free_sig_from_sample(sample_index::Int, F::FreeTruncSigAlgMultiv)
  s = gens_in_shape(F)
  trunc_level = truncation_level(F)
  return sum(s[i,sample_index] for i in (1:trunc_level)) + one(F)
end 


"""
    free_sig_bary(F::FreeTruncSigAlgMultiv)

Returns the barycenter element (`FreeTruncSigAlgMultivElem`) of the algebra `F`.

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
    y = free_sig_bary(F)
"""
function free_sig_bary(F::FreeTruncSigAlgMultiv)
  s = gens_in_shape(F)
  trunc_level = truncation_level(F)
  num_samp = number_of_samples(F)
  return sum(s[i,num_samp+1] for i in (1:trunc_level)) + one(F)
end 

"""
    gens(F::FreeTruncSigAlgMultiv)

Returns the generators of the free algebra as `FreeTruncSigAlgMultivElem`s.

# Example
    F, s = free_trunc_sig_alg_multiv(3, 2)
    G = gens(F)
"""
gens

function Oscar.gens(F::FreeTruncSigAlgMultiv)
  free_alg = free_algebra(F)
  free_alg_gens = gens(free_alg)
  return [FreeTruncSigAlgMultivElem(F,si) for si in free_alg_gens]
end 


"""
    gens_in_shape(F::FreeTruncSigAlgMultiv)

Returns the **generators of a free truncated multivariate signature algebra** `F` arranged 
in a 2-dimensional array according to truncation level and number of samples.

If the algebra includes a barycenter (`F.with_bary = true`), the extra barycenter element 
is added as an additional column.

# Arguments
- `F::FreeTruncSigAlgMultiv`: the free truncated multivariate signature algebra.

# Returns
- `Array{FreeTruncSigAlgMultivElem}`: a 2D array of generators with shape `(truncation_level, number_of_samples (+1 if bary)))`.

# Example
F, s = free_trunc_sig_alg_multiv(3, 2)  # trunc_level=3, 2 samples
G = gens_in_shape(F)  # returns a 3x3 array if with_bary=true
"""
function gens_in_shape(F::FreeTruncSigAlgMultiv)
  a1 = truncation_level(F)
  a2 = number_of_samples(F)
  if F.with_bary
    a2=a2+1
  end 
  return reshape(gens(F),(a1,a2))
end



#############################
# arithmetic
#############################

function Base.:*(a::FreeTruncSigAlgMultivElem,
                 b::FreeTruncSigAlgMultivElem)
  A = parent(a)
  trunc_level = truncation_level(A)
  return FreeTruncSigAlgMultivElem(A,trunc(a.elem * b.elem,trunc_level))
end

function Base.:*(a::FieldElem,
                 b::FreeTruncSigAlgMultivElem)
  A = parent(b)
  trunc_level = truncation_level(A)
  return FreeTruncSigAlgMultivElem(A,trunc(a * b.elem,trunc_level))
end

function Base.:+(a::FreeTruncSigAlgMultivElem,
                 b::FreeTruncSigAlgMultivElem)
  A = parent(a)
  trunc_level = truncation_level(A)
  return FreeTruncSigAlgMultivElem(A,trunc(a.elem + b.elem,trunc_level))
end

function Base.:-(a::FreeTruncSigAlgMultivElem,
                 b::FreeTruncSigAlgMultivElem)
  A = parent(a)
  trunc_level = truncation_level(A)
  return FreeTruncSigAlgMultivElem(A,trunc(a.elem - b.elem,trunc_level))
end

  
function Base.:^(f::FreeTruncSigAlgMultivElem,n::Int)
  res = f
  for i in (2:n)
    res = res*f
  end
  return res
end

function Oscar.exp(f::FreeTruncSigAlgMultivElem)
  A = parent(f)
  trunc_level = truncation_level(A)
  res = f
  for ell in (2:trunc_level)
    res += QQ(1,factorial(ell))*f^ell
  end
  return res + one(A)
end

function Oscar.log(f::FreeTruncSigAlgMultivElem)
  A = parent(f)
  trunc_level = truncation_level(A)
  res = zero(A)
  g = f-one(A)
  for ell in (1:trunc_level)
    res += QQ((-1)^(ell+1),ell)*g^ell
  end
  return res
end

function Oscar.inv(f::FreeTruncSigAlgMultivElem)
  log_f = log(f)
  return exp(QQ(-1,1)*log_f)
end


##################
# barycenter constructors 
##################
"""
    free_barycenter_2samples(trunc_level::Int)

Computes the barycenter of 2 sample elements in the free truncated algebra.

# Example
    bary = free_barycenter_2samples(3)
"""
function free_barycenter_2samples(_trunc_level::Int)
  F,s = free_trunc_sig_alg_multiv(_trunc_level,2)
  x = free_sig_from_sample(1,F)
  y = free_sig_from_sample(2,F)
  return x*exp(QQ(1,2)*log(inv(x)*y))
end

"""
    bary_defining_polynomial_system(trunc_level::Int, num_samples::Int)

Returns the polynomial system defining the barycenter of `num_samples` in the algebra.

# Example
    poly_sys = bary_defining_polynomial_system(3, 2)
"""
function bary_defining_polynomial_system(_trunc_level::Int,_num_samples::Int)
  F,s = free_trunc_sig_alg_multiv(_trunc_level,_num_samples,true) # with_bary = true
  x = [free_sig_from_sample(i,F) for i in (1:_num_samples)]
  y = free_sig_bary(F)
  return QQ(1,_num_samples)*sum(log(inv(y)*xi) for xi in x)
end 
  
