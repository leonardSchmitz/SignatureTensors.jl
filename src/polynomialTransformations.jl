################################################################################
#
#  Integration
#
################################################################################

function exponent_vectors(a::MPolyRingElem{T}; inplace::Bool = false) where T <: RingElement
   return Generic.MPolyExponentVectors(a, inplace=inplace)
end

function exponent_vectors(::Type{Vector{S}}, a::MPolyRingElem{T}; inplace::Bool = false) where {T <: RingElement, S}
   return Generic.MPolyExponentVectors(Vector{S}, a, inplace=inplace)
end

function integration(f::MPolyRingElem{T}, j::Int) where T <: RingElement
   R = parent(f)
   iterz = zip(coefficients(f), exponent_vectors(f))
   Ctx = Generic.MPolyBuildCtx(R)
   for (c, v) in iterz
      if v[j] >= 1
         prod = QQ(1,v[j])*c
         v[j] += 1
         push_term!(Ctx, prod, v)
      end
   end
   return finish(Ctx)
end

function integration(f::MPolyRingElem{T}, x::MPolyRingElem{T}) where T <: RingElement
   return integration(f, var_index(x))
end
