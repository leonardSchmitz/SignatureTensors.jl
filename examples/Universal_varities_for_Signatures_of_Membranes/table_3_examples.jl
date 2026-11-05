using Oscar, SignatureTensors
using Dates

function log_time(path::AbstractString, tag::AbstractString, msg::AbstractString)
    open(path, "a") do io
        println(io, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", tag, " | ", msg)
    end
end

fmt(t) = string(round(t, digits=3), " s")

function sig_flatten_top(AS, k::Int)
    nperm = factorial(k)
    return reduce(vcat, [vec(AS[ntuple(_ -> Colon(), k)..., ν]) for ν in 1:nperm])
end

function ambient_dimension(S_flat)
    mon_index = Dict{Any,Int}()
    next_idx = Ref(0)
    for f in S_flat
        for m in monomials(f)
            if !haskey(mon_index, m)
                next_idx[] += 1
                mon_index[m] = next_idx[]
            end
        end
    end
    ncols = next_idx[]
    nrows = length(S_flat)
    M = zero_matrix(QQ, nrows, ncols)
    for (i, f) in enumerate(S_flat)
        for (c, m) in zip(coefficients(f), monomials(f))
            M[i, mon_index[m]] = c
        end
    end
    return rank(M) - 1
end

function generators_by_degree(Ig)
    mg = minimal_generating_set(Ig)
    bins = Dict{Int,Int}()
    for g in mg
        dg = total_degree(g)
        bins[dg] = get(bins, dg, 0) + 1
    end
    return bins
end

gens_string(bins::Dict{Int,Int}) = isempty(bins) ? "0" :
    join([string(bins[k]) for k in sort(collect(keys(bins)))], ",")

function image_degree(R, S_flat, tag, timing_path, times)
    N = length(S_flat)

    log_time(timing_path, tag, "KERNEL: starting (ambient N=$N, params=$(ngens(R)))")
    t = time()
    S, _ = polynomial_ring(QQ, :x => 1:N)
    I = kernel(hom(S, R, S_flat))
    times[:kernel] = time() - t
    log_time(timing_path, tag, "KERNEL: $(fmt(times[:kernel]))  ($(ngens(I)) generators)")

    log_time(timing_path, tag, "REGRADE: starting")
    t = time()
    Sg, y = graded_polynomial_ring(QQ, :x => 1:N)
    remap = hom(S, Sg, y)
    Ig    = ideal(Sg, [remap(g) for g in gens(I)])
    @assert all(is_homogeneous, gens(Ig)) "Ig is not homogeneous: check that S_flat is homogeneous of degree k"
    Q, _  = quo(Sg, Ig)
    times[:regrade] = time() - t
    log_time(timing_path, tag, "REGRADE: $(fmt(times[:regrade]))")

    log_time(timing_path, tag, "DEGREE: starting")
    t = time()
    deg  = degree(Q)
    cdim = dim(Q)
    times[:degree] = time() - t
    log_time(timing_path, tag, "DEGREE: $(fmt(times[:degree]))  -> degree=$deg  dim(cone)=$cdim")

    return deg, cdim, Ig
end

function compute_entry(d::Int, k::Int, m::Int, n::Int; type::Symbol=:axis, timing_path::AbstractString)
    @assert type in (:axis, :mono) "type debe ser :axis (piecewise bilinear) o :mono (polinomial)"

    tag = "d=$d k=$k m=$m n=$n type=$type"
    times = Dict{Symbol,Float64}(:signature => 0.0, :product => 0.0, :flatten => 0.0,
                                  :ambient => 0.0, :kernel => 0.0, :regrade => 0.0,
                                  :degree => 0.0, :gens => 0.0)
    t0 = time()
    log_time(timing_path, tag, "--- START (m=$m, n=$n, type=$type) ---")

    t = time()
    R, a = polynomial_ring(QQ, :a => (1:d, 1:(m*n)))
    T = TruncatedTensorAlgebra(R, m*n, k, sequence_type=:p2)
    Cbase = sig(T, type, shape=(m, n))
    times[:signature] = time() - t

    t = time()
    AS = a * Cbase
    dim_map = factorial(k) * d^k
    times[:product] = time() - t

    t = time()
    S_flat = sig_flatten_top(AS, k)
    @assert length(S_flat) == dim_map "length(S_flat)=$(length(S_flat)) != dim_map=$dim_map"
    times[:flatten] = time() - t
    log_time(timing_path, tag, "FLATTEN: $(fmt(times[:flatten]))")

    log_time(timing_path, tag, "AMBIENT DIM (a): starting")
    t = time()
    amb = ambient_dimension(S_flat)
    times[:ambient] = time() - t
    log_time(timing_path, tag, "AMBIENT DIM (a): $(fmt(times[:ambient]))  -> a=$amb")

    deg, cdim, Ig = image_degree(R, S_flat, tag, timing_path, times)

    log_time(timing_path, tag, "GENS: starting")
    t = time()
    bins = generators_by_degree(Ig)
    gstr = gens_string(bins)
    times[:gens] = time() - t
    log_time(timing_path, tag, "GENS: $(fmt(times[:gens]))  -> gens=$gstr")

    times[:total] = time() - t0
    log_time(timing_path, tag,
        "--- END (m=$m,n=$n,type=$type): a=$amb dim=$cdim deg=$deg gens=$gstr  TOTAL: $(fmt(times[:total])) ---")

    return (a=amb, dim=cdim, deg=deg, gens=gstr, gens_bins=bins, times=times)
end

function save_entry_file(d::Int, k::Int, m::Int, n::Int, type::Symbol, res::NamedTuple; dir::AbstractString=".")
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    fname = "k$(k)_d$(d)_m$(m)_n$(n)_$(type)_$(ts).txt"
    path = joinpath(dir, fname)
    g(s) = round(get(res.times, s, 0.0), digits=3)
    open(path, "w") do io
        println(io, "k = ", k)
        println(io, "d = ", d)
        println(io, "m = ", m)
        println(io, "n = ", n)
        println(io, "type = ", type)
        println(io, "timestamp = ", ts)
        println(io, "a = ", res.a)
        println(io, "dim = ", res.dim)
        println(io, "deg = ", res.deg)
        println(io, "gens = ", res.gens)
        println(io, "total_time_s = ", g(:total))
        println(io, "signature_time_s = ", g(:signature))
        println(io, "product_time_s = ", g(:product))
        println(io, "flatten_time_s = ", g(:flatten))
        println(io, "ambient_time_s = ", g(:ambient))
        println(io, "kernel_time_s = ", g(:kernel))
        println(io, "regrade_time_s = ", g(:regrade))
        println(io, "degree_time_s = ", g(:degree))
        println(io, "gens_time_s = ", g(:gens))
    end
    return path
end

function compute_one(d::Int, k::Int, m::Int, n::Int; type::Symbol=:axis, log_dir::AbstractString=".")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    timing_path = joinpath(log_dir, "timings_single_k$(k)_d$(d)_m$(m)_n$(n)_$(type)_$(timestamp).txt")
    open(timing_path, "w") do io
        println(io, "# single entry k=$k d=$d m=$m n=$n type=$type")
        println(io, "# created: $(now())")
    end

    res = compute_entry(d, k, m, n; type=type, timing_path=timing_path)
    entry_path = save_entry_file(d, k, m, n, type, res; dir=log_dir)
    println("k=$k d=$d m=$m n=$n type=$type -> a=$(res.a) dim=$(res.dim) deg=$(res.deg) gens=$(res.gens)  (total $(fmt(res.times[:total])))")
    println("Guardado en: $entry_path")
    return res, entry_path
end

function run_afs_entries(d::Int, k::Int, mmax::Int, nmax::Int=mmax; type::Symbol=:axis, dir::AbstractString=".")
    timestamp   = Dates.format(now(), "yyyymmdd_HHMMSS")
    timing_path = joinpath(dir, "timings_k$(k)_d$(d)_m$(mmax)_n$(nmax)_$(type)_$(timestamp).txt")
    open(timing_path, "w") do io
        println(io, "# k=$k d=$d mmax=$mmax nmax=$nmax type=$type (solo m<=n, por simetria)")
        println(io, "# created: $(now())")
    end

    pairs = [(m, n) for m in 1:mmax for n in m:nmax]
    sort!(pairs, by = p -> (p[1]*p[2], p[1]))

    entry_paths = String[]
    t_all = time()
    for (idx, (m, n)) in enumerate(pairs)
        tag = "d=$d k=$k m=$m n=$n type=$type"
        log_time(timing_path, "GLOBAL", "combination $idx/$(length(pairs)): (m=$m, n=$n)")
        try
            res = compute_entry(d, k, m, n; type=type, timing_path=timing_path)
            entry_path = save_entry_file(d, k, m, n, type, res; dir=dir)
            push!(entry_paths, entry_path)
            log_time(timing_path, "RESULT", "m=$m n=$n a=$(res.a) dim=$(res.dim) deg=$(res.deg) gens=$(res.gens) -> $entry_path")
        catch e
            msg = sprint(showerror, e, catch_backtrace())
            log_time(timing_path, tag, "--- ERROR: $(first(msg, 3000)) ---")
            println("ERROR at (m=$m, n=$n): ", first(msg, 1500))
            ts = Dates.format(now(), "yyyymmdd_HHMMSS")
            errpath = joinpath(dir, "k$(k)_d$(d)_m$(m)_n$(n)_$(type)_$(ts)_ERROR.txt")
            open(errpath, "w") do io
                println(io, "k = ", k); println(io, "d = ", d); println(io, "m = ", m)
                println(io, "n = ", n); println(io, "type = ", type); println(io, "timestamp = ", ts)
                println(io, "ERROR = ", first(msg, 3000))
            end
            push!(entry_paths, errpath)
        end
        GC.gc()
    end

    open(timing_path, "a") do io
        println(io)
        println(io, "# TOTAL TIME: $(fmt(time() - t_all))")
    end

    return entry_paths, timing_path
end

d    = 3
k    = 3
mmax = 7
nmax = 7

#entry_paths_axis, timing_path_axis = run_afs_entries(d, k, mmax, nmax; type=:axis)
#println("Archivos generados (axis): ", length(entry_paths_axis))

entry_paths_mono, timing_path_mono = run_afs_entries(d, k, mmax, nmax; type=:mono)
println("Archivos generados (mono): ", length(entry_paths_mono))
