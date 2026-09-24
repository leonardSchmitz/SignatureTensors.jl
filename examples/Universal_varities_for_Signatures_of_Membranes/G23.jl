using Oscar
using Printf

varnames = String[

    "s_1_1", "s_2_1",

    "s_11_12", "s_11_21", "s_12_12", "s_12_21",
    "s_21_12", "s_21_21", "s_22_12", "s_22_21",

    "s_111_123", "s_111_132", "s_111_213", "s_111_231", "s_111_312", "s_111_321",
    "s_112_123", "s_112_132", "s_112_213", "s_112_231", "s_112_312", "s_112_321",
    "s_121_123", "s_121_132", "s_121_213", "s_121_231", "s_121_312", "s_121_321",
    "s_122_123", "s_122_132", "s_122_213", "s_122_231", "s_122_312", "s_122_321",
    "s_211_123", "s_211_132", "s_211_213", "s_211_231", "s_211_312", "s_211_321",
    "s_212_123", "s_212_132", "s_212_213", "s_212_231", "s_212_312", "s_212_321",
    "s_221_123", "s_221_132", "s_221_213", "s_221_231", "s_221_312", "s_221_321",
    "s_222_123", "s_222_132", "s_222_213", "s_222_231", "s_222_312", "s_222_321",
]

weights = vcat( fill(1, 2), fill(2, 8), fill(3, 48) )

@assert length(varnames) == 58
@assert length(weights) == 58


R, s = graded_polynomial_ring(QQ, varnames, weights)
V = Dict(string(g) => g for g in gens(R))

println("="^72)
println(" G_{2,3}: membrane signature ideal (Riffo-Schmitz-Amendola, Ex 4.1)")
println("="^72)

@printf "\n[1] Graded polynomial ring R = QQ[sigma^{<=3}]\n"
@printf "    number of variables : %d\n" ngens(R)
@printf "    weight distribution : level 1 -> 2 vars, level 2 -> 8 vars, level 3 -> 48 vars\n"



G23_gens = [

    V["s_1_1"]^2       - 2*V["s_11_12"] - 2*V["s_11_21"],
    V["s_1_1"]*V["s_2_1"] - V["s_12_12"] - V["s_12_21"] - V["s_21_12"] - V["s_21_21"],
    V["s_2_1"]^2       - 2*V["s_22_12"] - 2*V["s_22_21"],


    V["s_1_1"]*V["s_11_12"] - 3*V["s_111_123"] - 2*V["s_111_132"] - 2*V["s_111_213"] - V["s_111_231"] - V["s_111_312"],
    V["s_1_1"]*V["s_11_21"] - V["s_111_132"] - V["s_111_213"] - 2*V["s_111_231"] - 2*V["s_111_312"] - 3*V["s_111_321"],
    V["s_1_1"]*V["s_12_12"] - 2*V["s_112_123"] - V["s_112_132"] - 2*V["s_112_213"] - V["s_112_312"] - V["s_121_123"] - V["s_121_132"] - V["s_121_231"],
    V["s_1_1"]*V["s_12_21"] - V["s_112_132"] - 2*V["s_112_231"] - V["s_112_312"] - 2*V["s_112_321"] - V["s_121_213"] - V["s_121_312"] - V["s_121_321"],
    V["s_1_1"]*V["s_21_12"] - V["s_121_123"] - V["s_121_213"] - V["s_121_312"] - 2*V["s_211_123"] - 2*V["s_211_132"] - V["s_211_213"] - V["s_211_231"],
    V["s_1_1"]*V["s_21_21"] - V["s_121_132"] - V["s_121_231"] - V["s_121_321"] - V["s_211_213"] - V["s_211_231"] - 2*V["s_211_312"] - 2*V["s_211_321"],
    V["s_1_1"]*V["s_22_12"] - V["s_122_123"] - V["s_122_213"] - V["s_122_312"] - V["s_212_123"] - V["s_212_132"] - V["s_212_213"] - V["s_221_123"] - V["s_221_132"] - V["s_221_231"],
    V["s_1_1"]*V["s_22_21"] - V["s_122_132"] - V["s_122_231"] - V["s_122_321"] - V["s_212_231"] - V["s_212_312"] - V["s_212_321"] - V["s_221_213"] - V["s_221_312"] - V["s_221_321"],
    V["s_2_1"]*V["s_11_12"] - V["s_112_123"] - V["s_112_132"] - V["s_112_231"] - V["s_121_123"] - V["s_121_132"] - V["s_121_213"] - V["s_211_123"] - V["s_211_213"] - V["s_211_312"],
    V["s_2_1"]*V["s_11_21"] - V["s_112_213"] - V["s_112_312"] - V["s_112_321"] - V["s_121_231"] - V["s_121_312"] - V["s_121_321"] - V["s_211_132"] - V["s_211_231"] - V["s_211_321"],
    V["s_2_1"]*V["s_12_12"] - 2*V["s_122_123"] - 2*V["s_122_132"] - V["s_122_213"] - V["s_122_231"] - V["s_212_123"] - V["s_212_213"] - V["s_212_312"],
    V["s_2_1"]*V["s_12_21"] - V["s_122_213"] - V["s_122_231"] - 2*V["s_122_312"] - 2*V["s_122_321"] - V["s_212_132"] - V["s_212_231"] - V["s_212_321"],
    V["s_2_1"]*V["s_21_12"] - V["s_212_123"] - V["s_212_132"] - V["s_212_231"] - 2*V["s_221_123"] - V["s_221_132"] - 2*V["s_221_213"] - V["s_221_312"],
    V["s_2_1"]*V["s_21_21"] - V["s_212_213"] - V["s_212_312"] - V["s_212_321"] - V["s_221_132"] - 2*V["s_221_231"] - V["s_221_312"] - 2*V["s_221_321"],
    V["s_2_1"]*V["s_22_12"] - 3*V["s_222_123"] - 2*V["s_222_132"] - 2*V["s_222_213"] - V["s_222_231"] - V["s_222_312"],
    V["s_2_1"]*V["s_22_21"] - V["s_222_132"] - V["s_222_213"] - 2*V["s_222_231"] - 2*V["s_222_312"] - 3*V["s_222_321"],
]

G23 = ideal(R, G23_gens)

@printf "\n[2] Ideal G_{2,3}\n"
@printf "    number of explicit shuffle generators : %d\n" length(G23_gens)
@printf "    weight-2 (type (1,1)) generators      : 3\n"
@printf "    weight-3 (type (1,2)) generators      : 16\n"


all_hom = all(is_homogeneous, G23_gens)
ideal_hom = all(is_homogeneous, gens(G23))


@printf "\n[3] Homogeneity in the weight grading\n"
@printf "    every generator homogeneous : %s\n" all_hom
@printf "    ideal homogeneous           : %s\n" ideal_hom


@printf "\n[4] Algebraic invariants\n"

@printf "    computing Krull dimension ...\n"
t0 = time_ns(); d = dim(G23); t_dim = (time_ns() - t0) / 1e9
@printf "    dim(G23)                    = %d   (expected: 41 = g_1 + g_2 + g_3)   [%.2fs]\n" d t_dim

@printf "    computing codimension ...\n"
t0 = time_ns(); c = codim(G23); t_co = (time_ns() - t0) / 1e9
@printf "    codim(G23)                  = %d   (expected: 17)                   [%.2fs]\n" c t_co

@printf "    computing minimal generating set ...\n"
t0 = time_ns(); mgs = minimal_generating_set(G23); t_mg = (time_ns() - t0) / 1e9
@printf "    length(minimal_gen_set)     = %d   (expected: 19)                   [%.2fs]\n" length(mgs) t_mg

@printf "    testing primality ...\n"
t0 = time_ns(); pr = is_prime(G23); t_pr = (time_ns() - t0) / 1e9
@printf "    is_prime(G23)               = %s   (expected: true)                 [%.2fs]\n" pr t_pr

g1, g2, g3 = 2, 5, 34
G_3 = g1 + g2 + g3

@printf "\n[5] Consistency with the two-parameter Witt formula\n"
@printf "    g_1(2) = %d\n" g1
@printf "    g_2(2) = %d\n" g2
@printf "    g_3(2) = %d\n" g3
@printf "    G_3    = g_1 + g_2 + g_3 = %d\n" G_3
@printf "    dim V(G_{2,3})           = %d\n" d
@printf "    match?                   : %s\n" (d == G_3)


theta_11_w12 = V["s_12_12"] + V["s_12_21"] + V["s_21_12"] + V["s_21_21"]
theta_2_w1   = V["s_11_12"] + V["s_11_21"]
theta_2_w2   = V["s_22_12"] + V["s_22_21"]

rel_3_4 = theta_11_w12^2 - 4 * theta_2_w1 * theta_2_w2

@printf "\n[6] Example 3.4 sanity check\n"
@printf "    Theta((1,1), w12)^2 - 4 Theta(2, w1) Theta(2, w2) in G23 ?  %s\n" (rel_3_4 in G23)


println("\n" * "="^72)
@printf " Summary (d=2, k=3):  ambient dim = %d,  dim = %d,  codim = %d,  gens = %d\n" 58 d c length(mgs)
println("="^72)

