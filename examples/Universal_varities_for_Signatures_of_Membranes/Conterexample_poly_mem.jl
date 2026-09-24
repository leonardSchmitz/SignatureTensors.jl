using Oscar
using Random


const PERMS_S3 = ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1))

function moment_path_sig_level3(m::Int)
    D = Dict{NTuple{3,Int}, QQFieldElem}()
    for p1 in 1:m, p2 in 1:m, p3 in 1:m
        D[(p1,p2,p3)] = QQ(p2*p3, (p1+p2)*(p1+p2+p3))
    end
    return D
end


function axis_path_sig_level3(m::Int)
    D = Dict{NTuple{3,Int}, QQFieldElem}()
    for p1 in 1:m, p2 in 1:m, p3 in 1:m
        if p1 <= p2 <= p3
            counts = Dict{Int,Int}()
            for p in (p1,p2,p3)
                counts[p] = get(counts, p, 0) + 1
            end
            num_perms = 6
            for (_, c) in counts
                num_perms = div(num_perms, factorial(c))
            end
            D[(p1,p2,p3)] = QQ(num_perms, 6)
        else
            D[(p1,p2,p3)] = QQ(0)
        end
    end
    return D
end

function membrane_sig_level3(sig_m::Dict, sig_n::Dict, m::Int, n::Int)
    p = m * n
    result = Vector{Array{QQFieldElem,3}}(undef, 6)
    for t_idx in 1:6
        tau = PERMS_S3[t_idx]
        tau_inv = Vector{Int}(undef, 3)
        for j in 1:3
            tau_inv[tau[j]] = j
        end
        arr = Array{QQFieldElem,3}(undef, p, p, p)
        for i in 1:p, j in 1:p, k in 1:p
            arr[i,j,k] = zero(QQ)
        end
        for p1 in 1:m, q1 in 1:n, p2 in 1:m, q2 in 1:n, p3 in 1:m, q3 in 1:n
            j1 = (p1 - 1)*n + q1
            j2 = (p2 - 1)*n + q2
            j3 = (p3 - 1)*n + q3
            qs = (q1, q2, q3)
            q_perm = (qs[tau_inv[1]], qs[tau_inv[2]], qs[tau_inv[3]])
            arr[j1, j2, j3] = sig_m[(p1,p2,p3)] * sig_n[q_perm]
        end
        result[t_idx] = arr
    end
    return result
end

moment_membrane_sig_level3(m::Int, n::Int) =
    membrane_sig_level3(moment_path_sig_level3(m), moment_path_sig_level3(n), m, n)

axis_membrane_sig_level3(m::Int, n::Int) =
    membrane_sig_level3(axis_path_sig_level3(m), axis_path_sig_level3(n), m, n)

function signature_ideal(d::Int, m::Int, n::Int; family::Symbol=:moment)
    p = m * n
    sig = family == :moment ? moment_membrane_sig_level3(m, n) :
          family == :axis   ? axis_membrane_sig_level3(m, n)   :
          error("family must be :moment or :axis")

     A_names = String[]
    for i in 1:d, j in 1:p
        push!(A_names, "A_$(i)_$(j)")
    end
    A_ring, A_gens_flat = polynomial_ring(QQ, A_names)
    A = reshape(A_gens_flat, d, p)

     S_names = String[]
    for i1 in 1:d, i2 in 1:d, i3 in 1:d, t_idx in 1:6
        push!(S_names, "S_$(i1)_$(i2)_$(i3)_$(t_idx)")
    end
    S_ring, S_gens = graded_polynomial_ring(QQ, S_names)

     images = elem_type(A_ring)[]
    for i1 in 1:d, i2 in 1:d, i3 in 1:d, t_idx in 1:6
        s = zero(A_ring)
        arr = sig[t_idx]
        for j1 in 1:p, j2 in 1:p, j3 in 1:p
            c = arr[j1, j2, j3]
            iszero(c) && continue
            s += c * A[i1, j1] * A[i2, j2] * A[i3, j3]
        end
        push!(images, s)
    end

    S_ring_ug, S_gens_ug = polynomial_ring(QQ, S_names)
    phi_ug = hom(S_ring_ug, A_ring, images)
    I_ug = kernel(phi_ug)

    move = hom(S_ring_ug, S_ring, S_gens)
    I = ideal(S_ring, [move(g) for g in gens(I_ug)])
    return I, S_ring
end

function variety_invariants(I::MPolyIdeal)
    d_krull = dim(I)
    c = codim(I)
    deg = degree(I)
    G = minimal_generating_set(I)
    deg_hist = Dict{Int,Int}()
    for g in G
        d_ = total_degree(g)
        deg_hist[d_] = get(deg_hist, d_, 0) + 1
    end
    return (krull_dim=d_krull, codim=c, degree=deg, ngens=length(G), degrees=deg_hist)
end

function dim_via_jacobian(d::Int, m::Int, n::Int;
                          family::Symbol=:moment, prime::Int=32003, seed::Int=1)
    p_size = m * n
    sig = family == :moment ? moment_membrane_sig_level3(m, n) :
          family == :axis   ? axis_membrane_sig_level3(m, n)   :
          error("family must be :moment or :axis")

    Fp = GF(prime)
    rng = Random.MersenneTwister(seed)
    A = [Fp(rand(rng, 1:prime-1)) for _ in 1:d, _ in 1:p_size]

    num_out = d^3 * 6
    num_in = d * p_size
    J = zero_matrix(Fp, num_out, num_in)

    row = 0
    for i1 in 1:d, i2 in 1:d, i3 in 1:d, t_idx in 1:6
        row += 1
        arr = sig[t_idx]
        for j1 in 1:p_size, j2 in 1:p_size, j3 in 1:p_size
            c_qq = arr[j1, j2, j3]
            iszero(c_qq) && continue
            c = Fp(numerator(c_qq)) * inv(Fp(denominator(c_qq)))
            factors = ((i1, j1), (i2, j2), (i3, j3))
            others = (A[i2, j2]*A[i3, j3],
                      A[i1, j1]*A[i3, j3],
                      A[i1, j1]*A[i2, j2])
            for pos in 1:3
                (idx_row, idx_col) = factors[pos]
                col = (idx_row - 1)*p_size + idx_col
                J[row, col] += c * others[pos]
            end
        end
    end
    return rank(J)
end

function decompose_ideals(d::Int, m::Int, n::Int)
    println("  computing I_P (moment)...")
    IP, R = signature_ideal(d, m, n; family=:moment)
    invP = variety_invariants(IP)
    println("    I_P: ", invP)

    println("  computing I_L (axis)...")
    IL, _ = signature_ideal(d, m, n; family=:axis)
    invL = variety_invariants(IL)
    println("    I_L: ", invL)

    println("  computing J = I_P cap I_L (intersect)...")
    J = intersect(IP, IL)
    invJ = variety_invariants(J)
    println("    J  : ", invJ)

    genP = minimal_generating_set(IP)
    genL = minimal_generating_set(IL)

    println("  separating extras...")
    extras_P = elem_type(R)[]
    for g in genP
        (g in J) || push!(extras_P, g)
    end
    extras_L = elem_type(R)[]
    for g in genL
        (g in J) || push!(extras_L, g)
    end

    return (IP=IP, IL=IL, J=J,
            invP=invP, invL=invL, invJ=invJ,
            extras_P=extras_P, extras_L=extras_L, ring=R)
end

dP_212 = dim_via_jacobian(2, 1, 2; family=:moment)
dL_212 = dim_via_jacobian(2, 1, 2; family=:axis)
println("  dim P (affine) = $dP_212")
println("  dim L (affine) = $dL_212")

res_212 = decompose_ideals(2, 1, 2)

println("\n" * "="^70)
println("Generator explícit")
println("="^70)

println("\n--- Generadores shared (J = I_P cap I_L) ---")
G_shared_212 = minimal_generating_set(res_212.J)
for (k, g) in enumerate(G_shared_212)
    println("  q$k = $g")
end

println("\n--- Extras I_P (moment / polynomial) ---")
for (k, g) in enumerate(res_212.extras_P)
    println("  pe$k = $g")
end

println("\n--- Extras  I_L (axis / piecewise bilinear) ---")
for (k, g) in enumerate(res_212.extras_L)
    println("  le$k = $g")
end

println("\n" * "="^70)
println("Summary (d,m,n)=(2,1,2), k=3")
println("="^70)
println("                    | P (moment)  |  L (axis)  |  J = P cap L")
println("  ------------------|-------------|------------|-------------")
println("  krull_dim (afín)  | ",
        lpad(res_212.invP.krull_dim, 11), " | ",
        lpad(res_212.invL.krull_dim, 10), " | ",
        lpad(res_212.invJ.krull_dim, 12))
println("  codim             | ",
        lpad(res_212.invP.codim, 11), " | ",
        lpad(res_212.invL.codim, 10), " | ",
        lpad(res_212.invJ.codim, 12))
println("  degree            | ",
        lpad(res_212.invP.degree, 11), " | ",
        lpad(res_212.invL.degree, 10), " | ",
        lpad(res_212.invJ.degree, 12))
println("  # min. generators | ",
        lpad(res_212.invP.ngens, 11), " | ",
        lpad(res_212.invL.ngens, 10), " | ",
        lpad(res_212.invJ.ngens, 12))
println("  degree histogram P: ", res_212.invP.degrees)
println("  degree histogram L: ", res_212.invL.degrees)
println("  degree histogram J: ", res_212.invJ.degrees)
