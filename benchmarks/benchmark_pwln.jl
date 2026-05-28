using SignatureTensors
using BenchmarkTools
using Oscar

function pair_table(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    @assert size(A) == size(B) "A and B must have the same dimensions"

    m, n = size(A)
    T = Array{Tuple{Int,Int}}(undef, m, n)

    for i in 1:m
        for j in 1:n
            T[i,j] = (floor(Int, 1000*A[i,j]), floor(Int, 1000*B[i,j]))
        end
    end

    return T
end

function latex_table(A, B; rowlabels=nothing, collabels=nothing)
    T = pair_table(A,B)
    m, n = size(T)

    io = IOBuffer()
    println(io, "\\begin{tabular}{|" * "c|"^(n+1) * "}")
    println(io, "\\hline")

    # Column labels
    if collabels !== nothing
        print(io, "\$d\\!\\setminus\\!m\$ ")
        for c in collabels
            print(io, " & $c")
        end
        println(io, " \\\\ \\hline")
    end

    # Rows
    for i in 1:m
        print(io, rowlabels === nothing ? i : rowlabels[i])

        for j in 1:n
            a, b = T[i,j]

            cell = if a < b
                "\\textbf{$a}, $b"
            elseif b < a
                "$a, \\textbf{$b}"
            else
                "$a, $b"
            end

            print(io, " & $cell")
        end

        println(io, " \\\\ \\hline")
    end

    println(io, "\\end{tabular}")

    return String(take!(io))
end


k = 3;
ds = Vector(10:10:60);
N = 5;
B_chen_3 = benchmark_signature(ds,ds,k,num_samples=N,algorithm=:Chen,FloatN=true)
B_cong_3= benchmark_signature(ds,ds,k,num_samples=N,algorithm=:congruence,FloatN=true)
print(latex_table(B_chen_3[1],B_cong_3[1], rowlabels=ds, collabels=ds))  # Table 1 (k=3)


k = 4;
B_chen = benchmark_signature(ds,ds,k,num_samples=N,algorithm=:Chen,FloatN=true)
B_cong= benchmark_signature(ds,ds,k,num_samples=N,algorithm=:congruence,FloatN=true)
print(latex_table(B_chen[1],B_cong[1], rowlabels=ds, collabels=ds))  # Table 1 (k=4)



k = 3;
ds = Vector(10:10:60);
N = 100;

P_cong_3= benchmark_signature(ds,ds,k,signature_path=:poly,num_samples=N,algorithm=:congruence,FloatN=true)

print(latex_table(B_cong_3[1],P_cong_3[1], rowlabels=ds, collabels=ds))  # Table 1 (k=3)


k = 4;

P_cong= benchmark_signature(ds,ds,k,num_samples=N,signature_path=:poly,algorithm=:congruence,FloatN=true)

print(latex_table(B_cong[1],P_cong[1], rowlabels=ds, collabels=ds))  # Table 1 (k=4)












