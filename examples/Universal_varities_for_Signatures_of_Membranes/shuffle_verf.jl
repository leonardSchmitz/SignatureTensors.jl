perms(n) = n == 0 ? [Int[]] : [insert!(copy(p), i, n) for p in perms(n - 1) for i in 1:n]

comp(a, b) = a[b]                       # (a∘b)(j) = a(b(j))

shiftconcat(t1, t2) = vcat(t1, length(t1) .+ t2)   # τ1 \righttriangle τ2

function Sh(t1, t2)
    k1, k2 = length(t1), length(t2)
    [nu for nu in perms(k1 + k2) if issorted(nu[t1]) && issorted(nu[k1 .+ t2])]
end

Sh(k1::Int, k2::Int) = Sh(collect(1:k1), collect(1:k2))

act(pi, w) = w[invperm(pi)]             # (\pji·w)_c = w_{\phi^{-1}(c)}

function addterm!(D, key)
    D[key] = get(D, key, 0) + 1
    return D
end

function shuffleA(w1, t1, w2, t2)
    k1, k2 = length(w1), length(w2)
    w = vcat(w1, w2)
    D = Dict{Tuple{Vector{Int},Vector{Int}},Int}()
    for pi in Sh(k1, k2), lam in Sh(t1, t2)
        addterm!(D, (act(pi, w), comp(lam, pi)))
    end
    D
end

function shuffleB(w1, t1, w2, t2; useinv = false)
    k1, k2 = length(w1), length(w2)
    w = vcat(w1, w2)
    T = shiftconcat(t1, t2)
    useinv && (T = invperm(T))
    D = Dict{Tuple{Vector{Int},Vector{Int}},Int}()
    for pi in Sh(k1, k2), lam in Sh(k1, k2)
        addterm!(D, (act(pi, w), comp(comp(lam, T), pi)))
    end
    D
end

words(d, k) = k == 0 ? [Int[]] : [vcat(w, [a]) for w in words(d, k - 1) for a in 1:d]

function compare(d, kmax; useinv = false, verbose = true)
    bad = 0
    total = 0
    for k1 in 1:kmax, k2 in 1:kmax
        k1 + k2 > kmax && continue
        for w1 in words(d, k1), w2 in words(d, k2),
            t1 in perms(k1), t2 in perms(k2)

            total += 1
            A = shuffleA(w1, t1, w2, t2)
            B = shuffleB(w1, t1, w2, t2; useinv = useinv)
            if A != B
                bad += 1
                if verbose && bad == 1
                    println("  first mismatch: (w1,τ1) = ($w1,$t1), (w2,τ2) = ($w2,$t2)")
                    for key in sort(collect(union(keys(A), keys(B))))
                        a, b = get(A, key, 0), get(B, key, 0)
                        a != b && println("    $key : A=$a  B=$b")
                    end
                end
            end
        end
    end
    println("d=$d, k1+k2≤$kmax, useinv=$useinv : $bad / $total pairs disagree")
end

compare(2, 5; useinv = false)
compare(2, 5; useinv = true)
