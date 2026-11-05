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

    return deg, cdim, ngens(I)
end

function image_degree_elim(R, S_flat, tag, timing_path, times)
    s = ngens(R)
    N = length(S_flat)

    log_time(timing_path, tag, "BUILD BIG RING: starting (a-vars=$s, x-vars=$N)")
    t = time()
    Rbig, av, xv = polynomial_ring(QQ, :a => 1:s, :x => 1:N)
    phi = hom(R, Rbig, av)
    Jideal = ideal(Rbig, [xv[j] - phi(S_flat[j]) for j in 1:N])
    times[:build] = time() - t
    log_time(timing_path, tag, "BUILD BIG RING: $(fmt(times[:build]))")

    log_time(timing_path, tag, "ELIMINATE: starting (eliminating $s a-vars)")
    t = time()
    Ielim = eliminate(Jideal, av)
    times[:eliminate] = time() - t
    log_time(timing_path, tag, "ELIMINATE: $(fmt(times[:eliminate]))  ($(ngens(Ielim)) generators)")

    log_time(timing_path, tag, "DEGREE: starting")
    t = time()
    Sg, y = graded_polynomial_ring(QQ, :x => 1:N)
    proj = hom(Rbig, Sg, vcat([zero(Sg) for _ in 1:s], y))
    Ig   = ideal(Sg, [proj(g) for g in gens(Ielim)])
    @assert all(is_homogeneous, gens(Ig)) "Ig is not homogeneous: check that S_flat is homogeneous of degree k"
    Q, _ = quo(Sg, Ig)
    deg  = degree(Q)
    cdim = dim(Q)
    times[:degree] = time() - t
    log_time(timing_path, tag, "DEGREE: $(fmt(times[:degree]))  -> degree=$deg  dim(cone)=$cdim")

    return deg, cdim, ngens(Ielim)
end

function compute_entry(d, m, n, k; timing_path, method::Symbol=:kernel)
    tag = "d=$d k=$k m=$m n=$n"
    times = Dict{Symbol,Float64}(:signature => 0.0, :product => 0.0, :flatten => 0.0,
                                 :kernel => 0.0, :regrade => 0.0, :degree => 0.0,
                                 :build => 0.0, :eliminate => 0.0)
    t0 = time()
    log_time(timing_path, tag, "--- START (m=$m, n=$n) ---")

    log_time(timing_path, tag, "SIGNATURE: starting")
    t = time()
    R, a = polynomial_ring(QQ, :a => (1:d, 1:(m*n)))
    T = TruncatedTensorAlgebra(R, m*n, k, sequence_type=:p2)
    Caxis_d = sig(T, :axis, shape=(m,n))
    times[:signature] = time() - t
    log_time(timing_path, tag, "SIGNATURE: $(fmt(times[:signature]))")

    log_time(timing_path, tag, "PRODUCT a*Caxis_d: starting")
    t = time()
    AS = a * Caxis_d
    dim_map = factorial(k) * d^k
    times[:product] = time() - t
    log_time(timing_path, tag, "PRODUCT: $(fmt(times[:product]))")

    t = time()
    S_flat = sig_flatten_top(AS, k)
    @assert length(S_flat) == dim_map "length(S_flat)=$(length(S_flat)) != dim_map=$dim_map"
    times[:flatten] = time() - t
    log_time(timing_path, tag, "FLATTEN: $(fmt(times[:flatten]))")

    deg, cdim, ng = method === :elim ?
        image_degree_elim(R, S_flat, tag, timing_path, times) :
        image_degree(R, S_flat, tag, timing_path, times)

    times[:total] = time() - t0
    log_time(timing_path, tag,
        "--- END (m=$m,n=$n): degree=$deg  dim(cone)=$cdim  ngens=$ng  TOTAL: $(fmt(times[:total])) ---")
    return deg, cdim, ng, times
end

function write_tex(path, table, d, k, mmax, nmax)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "\\begin{tabular}{c|", repeat("c", nmax), "}")
        println(io, "\\hline")
        println(io, "\$m \\backslash n\$ & ", join(string.(1:nmax), " & "), " \\\\")
        println(io, "\\hline")
        for m in 1:mmax
            cells = [n <= m ? (haskey(table, (m, n)) ? string(table[(m, n)]) : "--") : "" for n in 1:nmax]
            println(io, m, " & ", join(cells, " & "), " \\\\")
        end
        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\caption{Degrees of the signature variety: \$d=$(d)\$, \$k=$(k)\$}")
    end
    mv(tmp, path, force=true)
end

function append_result(path, m, n, deg, cdim, ng, tm)
    g(s) = round(get(tm, s, 0.0), digits=3)
    open(path, "a") do io
        println(io, m, " ", n, " ", deg, " ", cdim, " ", ng, " ",
                g(:total), " ", g(:signature), " ", g(:product), " ", g(:flatten), " ",
                g(:kernel), " ", g(:regrade), " ", g(:build), " ", g(:eliminate), " ", g(:degree))
        flush(io)
    end
end

function run_table(d::Int, k::Int, mmax::Int, nmax::Int=mmax; method::Symbol=:kernel)
    timestamp    = Dates.format(now(), "yyyymmdd_HHMMSS")
    timing_path  = "degtimings_$(method)_d$(d)_k$(k)_m$(mmax)_n$(nmax)_$(timestamp).txt"
    tex_path     = "degtable_$(method)_d$(d)_k$(k)_m$(mmax)_n$(nmax)_$(timestamp).tex"
    results_path = "degresults_$(method)_d$(d)_k$(k)_m$(mmax)_n$(nmax)_$(timestamp).txt"

    open(timing_path, "w") do io
        println(io, "# DEGREES  method=$method d=$d k=$k mmax=$mmax nmax=$nmax")
        println(io, "# created: $(now())")
    end

    open(results_path, "w") do io
        println(io, "# DEGREES  method=$method d=$d k=$k mmax=$mmax nmax=$nmax")
        println(io, "# created: $(now())")
        println(io, "# m n degree dim ngens total signature product flatten kernel regrade build eliminate degree_step")
    end

    pairs = [(m, n) for m in 1:mmax for n in 1:m]
    sort!(pairs, by = p -> (p[1]*p[2], p[1]))

    table  = Dict{Tuple{Int,Int},Any}()
    dims   = Dict{Tuple{Int,Int},Any}()
    stored = Dict{Tuple{Int,Int},Dict{Symbol,Float64}}()
    write_tex(tex_path, table, d, k, mmax, nmax)

    t_all = time()
    for (idx, (m, n)) in enumerate(pairs)
        tag = "d=$d k=$k m=$m n=$n"
        log_time(timing_path, "GLOBAL", "combination $idx/$(length(pairs)): (m=$m, n=$n)")
        val, cdim, ng = "ERR", "-", "-"
        tm = Dict{Symbol,Float64}()
        t0 = time()
        try
            val, cdim, ng, tm = compute_entry(d, m, n, k; timing_path=timing_path, method=method)
        catch e
            tm[:total] = time() - t0
            msg = sprint(showerror, e, catch_backtrace())
            log_time(timing_path, tag, "--- ERROR (after $(fmt(tm[:total]))): $(first(msg, 3000)) ---")
            println("ERROR at (m=$m, n=$n): ", first(msg, 1500))
        end
        table[(m, n)] = val
        dims[(m, n)]  = cdim
        stored[(m, n)] = tm
        write_tex(tex_path, table, d, k, mmax, nmax)
        append_result(results_path, m, n, val, cdim, ng, tm)
        log_time(timing_path, "RESULT",
            "m=$m n=$n degree=$val dim=$cdim ngens=$ng total_s=$(round(get(tm, :total, 0.0), digits=3))")
        GC.gc()
    end

    open(timing_path, "a") do io
        println(io)
        println(io, "# SUMMARY (seconds)")
        println(io, "# m n degree dim total signature product flatten kernel regrade build eliminate degree_step")
        for (m, n) in sort(collect(keys(stored)))
            tm = stored[(m, n)]
            g(s) = round(get(tm, s, 0.0), digits=3)
            println(io, m, " ", n, " ", table[(m, n)], " ", dims[(m, n)], " ", g(:total), " ",
                    g(:signature), " ", g(:product), " ", g(:flatten), " ",
                    g(:kernel), " ", g(:regrade), " ", g(:build), " ", g(:eliminate), " ", g(:degree))
        end
        println(io, "# TOTAL TABLE TIME: $(fmt(time() - t_all))")
    end

    return table, dims, stored, timing_path, tex_path, results_path
end

d      = 2
k      = 2
mmax   = 3
nmax   = 3
method = :elim  

table, dims, stored, timing_path, tex_path, results_path = run_table(d, k, mmax, nmax; method=method)

println("Timings: $timing_path")
println("TeX table: $tex_path")
println("Results: $results_path")
