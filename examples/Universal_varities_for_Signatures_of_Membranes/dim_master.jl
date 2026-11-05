# dimension_master.jl
#
# Computes dim P_{d,k,(m,n)} (dimension of the polynomial / piecewise bilinear
# membrane variety, Section 4) as the generic rank of the Jacobian of the
# parametrization A -> sigma^(k)(A * Dict^{m,n}), for Dict in {axis, mono}.

# Paper correspondence:
#   Table 1 (Example 5.9):  dictionary = :axis, k = 2, d = 2..12 (or as needed)
#   Table 2 (Example 6.7):  dictionary = :mono, k = 3, d in {2,3,4,5}
# (At k=2 the :axis and :mono dictionaries give the same variety by
#  Theorem 4.16, so :axis is used there because it is cheaper to evaluate;
#  at k=3 they differ, so Table 2 must use :mono, cf. Example 4.17.)

# Degrees and minimal-generator counts (Examples 4.1, 4.3, 4.17) are NOT
# computed here -- that needs the exact Groebner/elimination approach in
# degree_grob.jl, kept as a separate file.

using Oscar, SignatureTensors
using LinearAlgebra
using Dates


function log_time(path::AbstractString, tag::AbstractString, msg::AbstractString)
    open(path, "a") do io
        println(io, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", tag, " | ", msg)
    end
end

fmt(t) = string(round(t, digits=3), " s")


function random_point(nv::Int; bound=100)
    vals = filter(!iszero, -bound:bound)
    return [QQ(rand(vals)) for _ in 1:nv]
end

function random_point_rational(nv::Int; bound=5, maxden=100)
    pt = QQFieldElem[]
    for _ in 1:nv
        b = rand(1:maxden)
        a = 0
        while a == 0
            a = rand(-bound*b:bound*b)
        end
        push!(pt, QQ(a, b))
    end
    return pt
end

function exp_vectors(f)
    if isdefined(Oscar, :exponents)
        return Oscar.exponents(f)
    else
        return Oscar.AbstractAlgebra.exponent_vectors(f)
    end
end

function extract_terms(S_flat)
    col  = Int[]
    cf   = QQFieldElem[]
    vars = Vector{Vector{Int}}()
    exps = Vector{Vector{Int}}()
    for (j, f) in enumerate(S_flat)
        for (c, e) in zip(coefficients(f), exp_vectors(f))
            nz = findall(!iszero, e)
            isempty(nz) && continue
            push!(col, j)
            push!(cf, QQ(c))
            push!(vars, nz)
            push!(exps, e[nz])
        end
    end
    return (col=col, cf=cf, vars=vars, exps=exps)
end

function jacobian_from_terms(terms, pt, nv::Int, ncols::Int)
    col, cf, vars, exps = terms.col, terms.cf, terms.vars, terms.exps
    J = zero_matrix(QQ, nv, ncols)
    for k in eachindex(col)
        vs, es = vars[k], exps[k]
        for a in eachindex(vs)
            t = cf[k] * es[a]
            for b in eachindex(vs)
                ex = (a == b) ? es[b] - 1 : es[b]
                ex > 0 && (t *= pt[vs[b]]^ex)
            end
            J[vs[a], col[k]] += t
        end
    end
    return J
end


function sig_flatten_top(AS, k::Int)
    nperm = factorial(k)
    return reduce(vcat, [vec(AS[ntuple(_ -> Colon(), k)..., ν]) for ν in 1:nperm])
end

function compute_entry(d::Int, m::Int, n::Int, k::Int; dict::Symbol=:axis,
                        trials::Int=2, bound::Int=100, rational::Bool=false,
                        timing_path::AbstractString="timings.txt")
    @assert dict in (:axis, :mono) "dict must be :axis or :mono"

    tag = "dict=$dict d=$d k=$k m=$m n=$n"
    times = Dict{Symbol,Float64}(:signature => 0.0, :product => 0.0, :flatten => 0.0,
                                  :extract => 0.0, :jacobian => 0.0, :rank => 0.0)
    t0 = time()
    log_time(timing_path, tag, "--- START ---")

    t = time()
    R, a = polynomial_ring(QQ, :a => (1:d, 1:(m*n)))
    T = TruncatedTensorAlgebra(R, m*n, k, sequence_type=:p2)
    Cbase = sig(T, dict, shape=(m, n))
    times[:signature] = time() - t
    log_time(timing_path, tag, "SIGNATURE ($dict): $(fmt(times[:signature]))")

    t = time()
    AS = a * Cbase
    dim_map = factorial(k) * d^k
    times[:product] = time() - t
    log_time(timing_path, tag, "PRODUCT a*Cbase: $(fmt(times[:product]))")

    t = time()
    S_flat = sig_flatten_top(AS, k)
    @assert length(S_flat) == dim_map "length(S_flat)=$(length(S_flat)) != dim_map=$dim_map"
    times[:flatten] = time() - t
    log_time(timing_path, tag, "FLATTEN: $(fmt(times[:flatten]))")

    t = time()
    terms = extract_terms(S_flat)
    times[:extract] = time() - t
    log_time(timing_path, tag, "TERM EXTRACTION: $(fmt(times[:extract]))  ($(length(terms.col)) terms)")

    nv = ngens(R)
    ranks = Int[]
    for i in 1:trials
        pt = rational ? random_point_rational(nv, bound=bound) : random_point(nv, bound=bound)

        t = time()
        Jnum = jacobian_from_terms(terms, pt, nv, dim_map)
        dt = time() - t
        times[:jacobian] += dt

        t = time()
        r = rank(Jnum)
        dt = time() - t
        times[:rank] += dt
        log_time(timing_path, tag, "trial $i/$trials: rank = $r")
        push!(ranks, r)
    end

    val = maximum(ranks)
    times[:total] = time() - t0
    log_time(timing_path, tag, "--- END: rank = $val  ranks=$ranks  TOTAL: $(fmt(times[:total])) ---")
    return val, ranks, times
end


function write_tex(path, table, d, k, dict, mmax, nmax)
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
        println(io, "\\caption{\$\\dim \\mathcal P_{d,k,(m,n)}\$ (dict=$dict), \$d=$(d)\$, \$k=$(k)\$}")
    end
    mv(tmp, path, force=true)
end

function run_table(d::Int, k::Int, mmax::Int, nmax::Int=mmax; dict::Symbol=:axis,
                    trials::Int=2, bound::Int=100, rational::Bool=false,
                    out_dir::AbstractString=".")
    timestamp   = Dates.format(now(), "yyyymmdd_HHMMSS")
    timing_path = joinpath(out_dir, "timings_$(dict)_d$(d)_k$(k)_mmax$(mmax)_$(timestamp).txt")
    tex_path    = joinpath(out_dir, "table_$(dict)_d$(d)_k$(k)_mmax$(mmax)_$(timestamp).tex")

    open(timing_path, "w") do io
        println(io, "# dict=$dict d=$d k=$k mmax=$mmax nmax=$nmax trials=$trials bound=$bound rational=$rational")
        println(io, "# created: $(now())")
    end

    pairs = [(m, n) for m in 1:mmax for n in 1:min(m, nmax)]
    sort!(pairs, by = p -> (p[1]*p[2], p[1]))

    table  = Dict{Tuple{Int,Int},Any}()
    stored = Dict{Tuple{Int,Int},Dict{Symbol,Float64}}()
    write_tex(tex_path, table, d, k, dict, mmax, nmax)

    t_all = time()
    for (idx, (m, n)) in enumerate(pairs)
        tag = "dict=$dict d=$d k=$k m=$m n=$n"
        log_time(timing_path, "GLOBAL", "combination $idx/$(length(pairs)): (m=$m, n=$n)")
        val = "ERR"
        tm  = Dict{Symbol,Float64}()
        t0 = time()
        try
            val, _, tm = compute_entry(d, m, n, k; dict=dict, trials=trials, bound=bound,
                                        rational=rational, timing_path=timing_path)
        catch e
            tm[:total] = time() - t0
            msg = sprint(showerror, e, catch_backtrace())
            log_time(timing_path, tag, "--- ERROR (after $(fmt(tm[:total]))): $(first(msg, 3000)) ---")
            println("ERROR at (m=$m, n=$n): ", first(msg, 1500))
        end
        table[(m, n)]  = val
        stored[(m, n)] = tm
        write_tex(tex_path, table, d, k, dict, mmax, nmax)
        log_time(timing_path, "RESULT", "m=$m n=$n value=$val total_s=$(round(get(tm, :total, 0.0), digits=3))")
        GC.gc()
    end

    open(timing_path, "a") do io
        println(io)
        println(io, "# SUMMARY (seconds)")
        println(io, "# m n value total signature product flatten extract jacobian rank")
        for (m, n) in sort(collect(keys(stored)))
            tm = stored[(m, n)]
            g(s) = round(get(tm, s, 0.0), digits=3)
            println(io, m, " ", n, " ", table[(m, n)], " ", g(:total), " ", g(:signature), " ",
                    g(:product), " ", g(:flatten), " ", g(:extract), " ", g(:jacobian), " ", g(:rank))
        end
        println(io, "# TOTAL TABLE TIME: $(fmt(time() - t_all))")
    end

    return table, stored, timing_path, tex_path
end

function build_table1(dmax::Int=8; mmax::Int=dmax+1, kwargs...)
    results = Dict{Int,Any}()
    for d in 2:dmax
        results[d] = run_table(d, 2, mmax; dict=:axis, kwargs...)
    end
    return results
end

function build_table2(ds=(2,3,4,5); mmax::Int=7, kwargs...)
    results = Dict{Int,Any}()
    for d in ds
        results[d] = run_table(d, 3, mmax; dict=:mono, kwargs...)
    end
    return results
end


run_table(3, 2, 5; dict=:axis, trials=2, bound=100)
