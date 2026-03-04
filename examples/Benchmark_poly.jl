
using  BenchmarkTools, Random, LinearAlgebra, Statistics

# -------------------------------
# Benchmarking pwln signatures
# -------------------------------

# Parameters to test
ds = [2, 3, 4, 5]        # dimensions of the truncated tensor algebra
ms = [2, 3, 4, 5]      
k = 3                       # truncation level
num_matrices = 5            # number of random matrices per combination

# Dictionary to store timing results
results = Dict{Tuple{Int,Int}, Vector{Float64}}()

# Loop over all combinations of d and m
for d in ds
    for m in ms
        times = Float64[]           # store times for current combination
        T = TruncatedTensorAlgebra(QQ, d, k)  # create truncated tensor algebra
        
        for _ in 1:num_matrices
            # Generate a random coefficient matrix of size (d x m)
            A = QQ.(rand(-20:20, d, m))
            
            # Benchmark the computation of the pwln signature
            t = @belapsed sig($T, :pwln, coef=$A)
            
            # Store the elapsed time
            push!(times, t)
        end
        
        # Save all times for this combination of d and m
        results[(d,m)] = times
        
        # Print individual times in milliseconds
        println("d = $d, m = $m -> times (ms) = ", round.(times .* 1000, digits=2))
    end
end

# -------------------------------
# Optional: print average times
# -------------------------------
println("\nAverage times per combination (ms):")
for ((d,m), times) in sort(collect(results))
    avg_time = mean(times) * 1000
    println("d=$d, m=$m -> avg time = $(round(avg_time,digits=2)) ms")
end

