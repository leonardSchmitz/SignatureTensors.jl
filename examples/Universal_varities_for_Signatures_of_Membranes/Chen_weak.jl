using Oscar
using SignatureTensors
using Statistics
using LinearAlgebra

function _p2_permutations_ntuple(v::NTuple{N,Int}) where N
    N == 1 && return [v]
    out = NTuple{N,Int}[]
    for i in 1:N
        x = v[i]
        rest = ntuple(t -> v[t < i ? t : t + 1], N - 1)
        for p in _p2_permutations_ntuple(rest)
            push!(out, (x, p...))
        end
    end
    return out
end

_p2_permutations(j::Int) = _p2_permutations_ntuple(ntuple(identity, j))

function _perm_decomposition_points(nu::NTuple{K,Int}) where K
    pts = Int[0]
    running_max = 0
    for a in 1:K
        running_max = max(running_max, nu[a])
        running_max == a && push!(pts, a)
    end
    return pts
end


function _perm_split(nu::NTuple{K,Int}, a::Int) where K
    lo = ntuple(t -> nu[t], a)
    hi = ntuple(t -> nu[a + t] - a, K - a)
    return lo, hi
end


const _P2_PERM_POSITION = Dict{Int, Dict{Tuple{Vararg{Int}}, Int}}()

function _perm_position_p2(j::Int, nu)
    j == 0 && return 1
    table = get!(_P2_PERM_POSITION, j) do
        tbl = Dict{Tuple{Vararg{Int}}, Int}()
        for (r, p) in enumerate(_p2_permutations(j))
            tbl[Tuple(p)] = r
        end
        tbl
    end
    return table[Tuple(nu)]
end


function _p2_slice(x::TruncatedTensorAlgebraElem, j::Int, r::Int)
    seq = tensor_sequence(x)
    j == 0 && return seq[1]
    tj = seq[j + 1]
    ndims(tj) == j && return tj          # level 1 stored without trailing mode
    return tj[ntuple(_ -> Colon(), j)..., r]
end

function signature_product(ST_1::TruncatedTensorAlgebraElem{R,E},
                           ST_2::TruncatedTensorAlgebraElem{R,E}) where {R,E}

    p1, p2 = parent(ST_1), parent(ST_2)

    sequence_type(p1) == :p2 && sequence_type(p2) == :p2 ||
        throw(ArgumentError("signature_product requires sequence_type = :p2 on both factors"))
    base_dimension(p1) == base_dimension(p2) ||
        throw(ArgumentError("signature_product: ambient dimensions differ"))
    truncation_level(p1) == truncation_level(p2) ||
        throw(ArgumentError("signature_product: truncation levels differ"))

    k   = truncation_level(p1)
    d   = base_dimension(p1)
    alg = base_algebra(p1)

    out = Vector{Array{E}}(undef, k + 1)
    out[1] = fill(tensor_sequence(ST_1)[1][] * tensor_sequence(ST_2)[1][], ())

    for j in 1:k
        level = fill(zero(alg), ntuple(_ -> d, j)..., factorial(j))

        for (r, nu) in enumerate(_p2_permutations(j))
            acc = fill(zero(alg), ntuple(_ -> d, j)...)

            for split in _perm_decomposition_points(Tuple(nu))
                if split == 0
                    acc = acc .+ _p2_slice(ST_2, j, r)
                elseif split == j
                    acc = acc .+ _p2_slice(ST_1, j, r)
                else
                    lo, hi = _perm_split(Tuple(nu), split)
                    left  = _p2_slice(ST_1, split,     _perm_position_p2(split,     lo))
                    right = _p2_slice(ST_2, j - split, _perm_position_p2(j - split, hi))
                    acc = acc .+ concatenate_tensors_TA(left, right)
                end
            end

            level[ntuple(_ -> Colon(), j)..., r] = acc
        end

        out[j + 1] = level
    end

    return TruncatedTensorAlgebraElem{R,E}(p1, out)
end

function signature_product(STs::AbstractVector{<:TruncatedTensorAlgebraElem})
    isempty(STs) && error("signature_product of an empty list is undefined")
    result = STs[1]
    for i in 2:length(STs)
        result = signature_product(result, STs[i])
    end
    return result
end




function concat_pwbln_coef(Dx::AbstractArray{S,3}, Dy::AbstractArray{S,3}) where S
    m1, n1, d = size(Dx)
    m2, n2 = size(Dy, 1), size(Dy, 2)
    size(Dy, 3) == d || error("Dx and Dy must map into the same R^d")

    coef = fill(zero(Dx[1, 1, 1]), m1 + m2, n1 + n2, d)
    coef[1:m1, 1:n1, :] = Dx
    coef[(m1+1):(m1+m2), (n1+1):(n1+n2), :] = Dy
    return coef
end

function concat_pwbln_coef(Ax::AbstractMatrix{S}, mx::Int, nx::Int,
                           Ay::AbstractMatrix{S}, my::Int, ny::Int) where S
    d = size(Ax, 1)
    size(Ax, 2) == mx * nx || error("size(Ax,2) must equal mx*nx")
    size(Ay, 1) == d       || error("Ax and Ay must map into the same R^d")
    size(Ay, 2) == my * ny || error("size(Ay,2) must equal my*ny")

    Dx = Array{S}(undef, mx, nx, d)
    for i in 1:mx, j in 1:nx
        Dx[i, j, :] = Ax[:, (i - 1) * nx + j]
    end
    Dy = Array{S}(undef, my, ny, d)
    for i in 1:my, j in 1:ny
        Dy[i, j, :] = Ay[:, (i - 1) * ny + j]
    end

    Dz = concat_pwbln_coef(Dx, Dy)
    mz, nz = mx + my, nx + ny

    Az = Matrix{S}(undef, d, mz * nz)
    for i in 1:mz, j in 1:nz
        Az[:, (i - 1) * nz + j] = Dz[i, j, :]
    end

    return Az, (mz, nz)
end


d, k = 3, 3
T = TruncatedTensorAlgebra(QQ, d, k, sequence_type = :p2)

m1, n1 = 2, 2
m2, n2 = 2, 3

Dx = QQ.(reshape(1:(m1*n1*d), m1, n1, d))
Dy = QQ.(reshape(-1:-1:-(m2*n2*d), m2, n2, d))

ST_1 = sig(T, :pwbln, coef = Dx)
ST_2 = sig(T, :pwbln, coef = Dy)

ST = signature_product(ST_1, ST_2)

Dz = concat_pwbln_coef(Dx, Dy)
T_big = TruncatedTensorAlgebra(QQ, d, k, sequence_type = :p2)
ST_direct = sig(T_big, :pwbln, coef = Dz)

println("shapes match : ", size(Dz) == (m1+m2, n1+n2, d))
println("signature_product == direct construction : ",
        tensor_sequence(ST) == tensor_sequence(ST_direct))

# also test the matrix form of concat_pwbln_coef
Ax = reshape(permutedims(Dx, (3,2,1)), d, m1*n1)
Ay = reshape(permutedims(Dy, (3,2,1)), d, m2*n2)
Az, shape_z = concat_pwbln_coef(Ax, m1, n1, Ay, m2, n2)
ST_direct2 = sig(T_big, :pwbln, coef = Az, shape = shape_z)
println("matrix-form path agrees too : ",
        tensor_sequence(ST) == tensor_sequence(ST_direct2))


function weak_chen_chain(T, A::AbstractMatrix)
    d, l = size(A)
    sigmas = [sig(T, :pwbln, coef = reshape(A[:, i], 1, 1, d)) for i in 1:l]
    chain = Vector{TruncatedTensorAlgebraElem}(undef, l)
    chain[1] = sigmas[1]
    for j in 2:l
        chain[j] = signature_product(chain[j-1], sigmas[j])
    end
    return chain
end

 
 
function random_point(R; bound=5)
    return [rand(-bound:bound) for _ in gens(R)]
end
 
function evaluate_poly(f, pt)
    return evaluate(f, [QQ(v) for v in pt])
end
 
function evaluate_matrix(M, pt)
    return matrix(QQ, size(M,1), size(M,2),
        [evaluate_poly(M[i,j], pt) for i in 1:size(M,1), j in 1:size(M,2)])
end
 

function numerical_image_dim(J, R; trials=10, bound=100)
    ranks = Int[]
    for _ in 1:trials
        pt   = random_point(R, bound=bound)
        Jnum = evaluate_matrix(J, pt)
        push!(ranks, rank(Jnum))
    end
    println("Ranks: ", ranks)
    return maximum(ranks)
end

 
function chain_rank(chain, j, a, R; trials=10, bound=100)
    ST = chain[j]
    S_flat = vcat([vec(ST[:, :, :, r]) for r in 1:6]...)
    nrows = length(a)
    ranks = Int[]
    for _ in 1:trials
        pt = [QQ(v) for v in random_point(R, bound=bound)]
        J = zero_matrix(QQ, nrows, length(S_flat))
        for (col, f) in enumerate(S_flat), (row, ai) in enumerate(a)
            J[row, col] = evaluate(derivative(f, ai), pt)
        end
        push!(ranks, rank(J))
    end
    println("Ranks: ", ranks)
    return maximum(ranks)
end
 

d, l, k = 3, 20, 3
R, a = polynomial_ring(QQ, :a => (1:d, 1:l))
T = TruncatedTensorAlgebra(R, d, k, sequence_type = :p2)
chain = weak_chen_chain(T, a)
 
for i in 1:length(chain)
    println("Rank of chain[", i, "] = ", chain_rank(chain, i, a, R))
end
