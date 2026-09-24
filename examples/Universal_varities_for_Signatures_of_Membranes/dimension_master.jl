using Pkg
#Pkg.activate(expanduser("~/SignatureTensors"))
using Oscar, SignatureTensors

using Statistics
using LinearAlgebra


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

function sig_flatten_top(AS, k::Int)
    nperm = factorial(k)
    return reduce(vcat, [vec(AS[ntuple(_ -> Colon(), k)..., ν]) for ν in 1:nperm])
end

function axis_jacobian(d, m, n; k=4, trials=10, bound=100)

    GC.gc()

    R, a = polynomial_ring(QQ, :a => (1:d, 1:(m*n)))

    T = TruncatedTensorAlgebra(R, m*n, k, sequence_type=:p2)

    Caxis_d = sig(T, :axis, shape=(m,n))

    dim_map = factorial(k) * d^k        # k=3 -> 6*d^3 ;  k=4 -> 24*d^4

    AS = a * Caxis_d

    Caxis_d = nothing
    GC.gc()

    S_flat = sig_flatten_top(AS, k)     # longitud = dim_map

    @assert length(S_flat) == dim_map "length(S_flat)=$(length(S_flat)) != dim_map=$dim_map"

    J_Axis = zero_matrix(R, size(a,1) * size(a,2), dim_map)

    println("size(J_Axis) = ", size(J_Axis), "  (d=$d, m=$m, n=$n, k=$k)")

    for i1 in 1:size(a,1), i2 in 1:size(a,2), j in 1:dim_map
        J_Axis[(i1-1)*size(a,2) + i2, j] = derivative(S_flat[j], a[i1,i2])
    end

    # Camino Float64 (recomendado para k=4: matrices grandes, rango con
    # tolerancia relativa por defecto de LinearAlgebra.rank).
    ranks = Int[]
    for _ in 1:trials
        GC.gc()
        J_num = zeros(size(a,1)*size(a,2), dim_map)
        pt    = random_point(R, bound=bound)
        for i1 in 1:size(J_num,1), j1 in 1:size(J_num,2)
            J_num[i1, j1] = Float64(evaluate(J_Axis[i1, j1], pt))
        end
        push!(ranks, rank(J_num))
    end
    println("Ranks: ", ranks)
    return maximum(ranks)
end

function fill_table(d, mmax; k=4, trials=10, bound=1000)

    A = fill("", mmax, mmax)

    for m1 in 1:mmax
        for m2 in 1:mmax
            if m1 >= m2
                m = m1
                n = m2

                R, a = polynomial_ring(QQ, :a => (1:d, 1:(m*n)))
                T = TruncatedTensorAlgebra(R, m*n, k, sequence_type=:p2)
                Caxis_d = sig(T, :axis, shape=(m,n))

                AS = a * Caxis_d
                dim_map = factorial(k) * d^k

                S_flat = sig_flatten_top(AS, k)

                J = zero_matrix(R, size(a,1)*size(a,2), dim_map)
                for i1 in 1:size(a,1), i2 in 1:size(a,2), j in 1:dim_map
                    J[(i1-1)*size(a,2) + i2, j] = derivative(S_flat[j], a[i1,i2])
                end

                val = numerical_image_dim(J, R; trials=trials, bound=bound)
                A[m1, m2] = string(val)
            end
        end
    end

    return A
end

function print_latex_table(A, d, k)
    mmax = size(A,1)
    println("\\begin{tabular}{|c|" * "c"^mmax * "|}")
    println("\\hline")
    print("\$m_1 \\backslash m_2\$ & ")
    for j in 1:mmax
        print("$(j) & ")
    end
    println("\\\\")
    println("\\hline")
    for i in 1:mmax
        print("$(i) & ")
        for j in 1:mmax
            if i >= j
                print("$(A[i,j]) & ")
            else
                print(" & ")
            end
        end
        println("\\\\")
    end
    println("\\hline")
    println("\\end{tabular}")
    println("\\caption{\$d=$(d)\$, \$k=$(k)\$}")
end




axis_jacobian(2, 1, 1; k=2)


table = fill_table(3, 5; k=2)

#print_latex_table(table, 2, 4)

# axis_jacobian(3, 4, 4; k=4)


