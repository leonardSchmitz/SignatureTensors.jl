export 
    benchmark_signature

function benchmark_signature(ds::Vector{Int}, ms::Vector{Int}, k::Int; 
                             signature_path::Symbol=:pwln,
                             seq_type::Symbol=:iis,
                             num_samples::Int=100,algorithm::Symbol = :default,Seconds::Int=400)

    # Diccionarios para guardar resultados
    wall_time = Dict{Tuple{Int,Int}, Float64}()
    gc_time   = Dict{Tuple{Int,Int}, Float64}()
    memory    = Dict{Tuple{Int,Int}, Float64}()

    for d in ds
        for m in ms
            T = TruncatedTensorAlgebra(QQ, d, k, seq_type)
            A = QQ.(rand(-20:20, d, m))

            # Crear benchmarkable y correrlo
            benchset = @benchmarkable sig($T, $signature_path, coef=$A, algorithm=$algorithm) seconds=Seconds samples=num_samples
            bench = run(benchset)

            # Guardar medianos en nanosegundos / bytes
            wall_time[(d,m)] = median(bench).time / 1e9   
            gc_time[(d,m)]   = median(bench).gctime / 1e9 
            memory[(d,m)]    = median(bench).memory  / 1e6       
        end
    end


    n_ds = length(ds)
    n_ms = length(ms)
    wall_matrix = zeros(n_ds, n_ms)
    gc_matrix   = zeros(n_ds, n_ms)
    mem_matrix  = zeros(n_ds, n_ms)

    for ((d, m), t) in wall_time
        i = findfirst(==(d), ds)
        j = findfirst(==(m), ms)
        wall_matrix[i,j] = t
        gc_matrix[i,j]   = gc_time[(d,m)]
        mem_matrix[i,j]  = memory[(d,m)]
    end

    return wall_matrix, gc_matrix, mem_matrix
end

