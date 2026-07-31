# =====================================================================
#  TUTORIAL: Signature features vs. raw features for time series
#  classification, using SignatureTensors.jl (built on OSCAR).
#
#  Self-contained script - run from a fresh Julia session, nothing
#  else needs to be loaded first.
#
#  Usage:
#    cd ~/juliacon_demo
#    julia --project=.
#    julia> include("run_all_classifiers_multivariate.jl")
#
#  What it does:
#    For 4 multivariate datasets from the UEA/UCR archive
#    (BasicMotions, Cricket, ArticularyWordRecognition, NATOPS),
#    it builds two kinds of features per trace:
#      - "raw-flatten": just flatten the (time x channel) matrix
#        into one long vector.
#      - "signature":   compute the truncated path signature
#        (via SignatureTensors.jl) of the time-augmented path and
#        flatten that instead.
#    Then it trains 6 classifiers on each feature set and reports
#    test accuracy, so you can compare "with signature" vs.
#    "without signature" under identical classifiers.
#
#  Required packages (run once, in Pkg mode, if not already installed):
#    ] add Oscar SignatureTensors DecisionTree DelimitedFiles
#    ] add NearestNeighbors LIBSVM
#  (NearestNeighbors is not actually used below - k-NN was dropped
#   from the battery - but it's harmless to have installed.)
# =====================================================================

using Oscar
using SignatureTensors
using DecisionTree
using DelimitedFiles
using Statistics
using Random
using LinearAlgebra
using LIBSVM

# =====================================================================
# PART 1 - Signature core (SignatureTensors.jl)
# =====================================================================

"Time augmentation: (channel1,...,channelD) -> (t,channel1,...,channelD),
with t linearly spaced in [0,1]. Standard trick from Morrill et al.
(arXiv:2006.00873) that turns any path into one with a strictly
increasing coordinate, which helps the signature separate
reparametrizations of the same trace."
function time_augment(pts::Matrix{Float64})
    n = size(pts, 1)
    t = collect(range(0.0, 1.0; length = n))
    return hcat(t, pts)
end

"Builds the d x (n-1) matrix of per-segment increments (one column per
segment), which is the `coef` format expected by sig(T, :pwln; coef=...)
in SignatureTensors.jl. Columns are increments, NOT the raw vertices -
the package internally Chen-multiplies sig_segment_TA(T, coef[:,i])
over the columns."
function increments_matrix(pts::Matrix{Float64})
    D = diff(pts; dims = 1)
    return permutedims(D)
end

"Truncated tensor algebra over Float64, dimension d, truncation level k."
make_algebra(d::Int, k::Int) = TruncatedTensorAlgebra(Float64, d, k)

"Feature vector = vec(truncated signature) of one piecewise-linear path."
function signature_features(T, pts_augmented::Matrix{Float64})
    P = increments_matrix(pts_augmented)
    S = sig(T, :pwln; coef = P)
    return Float64.(vec(S))
end

"Signature features for a whole dataset (Vector of Matrix, each n x d).
`d` is inferred from the data; `augment` controls whether a time
channel is prepended before taking the signature. Output size grows
as d^k (with time augmentation, (d+1)^k), so keep k modest for large d."
function build_signature_matrix_general(paths::Vector{Matrix{Float64}}, k::Int; augment = true)
    d = size(paths[1], 2) + (augment ? 1 : 0)
    T = make_algebra(d, k)
    rows = Vector{Vector{Float64}}(undef, length(paths))
    for i in eachindex(paths)
        p = augment ? time_augment(paths[i]) : paths[i]
        rows[i] = signature_features(T, p)
    end
    return permutedims(reduce(hcat, rows))
end

"Drops near-constant columns (variance ~0), e.g. the level-0 constant
entry of the signature, or any signature coordinate that happens to
be identical across all samples in this particular dataset."
function drop_constant_columns(Xtr::Matrix, Xte::Matrix)
    keep = [Statistics.std(@view Xtr[:, j]) > 1e-12 for j in 1:size(Xtr, 2)]
    return Xtr[:, keep], Xte[:, keep]
end

"Random linear projection of the channel dimension (Johnson-Lindenstrauss
style scaling by 1/sqrt(d_old)). Useful to keep the signature's d^k
blow-up under control for high-dimensional inputs (e.g. NATOPS, d=24).
NOTE: in our experiments, projecting channels down before taking the
signature sometimes lost more information than it saved in compute -
see the discussion around NATOPS below. Prefer lowering k over
lowering d when you have the budget."
function random_projection(paths::Vector{Matrix{Float64}}, d_new::Int; seed = 0)
    d_old = size(paths[1], 2)
    Random.seed!(seed)
    P = randn(d_old, d_new) ./ sqrt(d_old)
    return [p * P for p in paths], P
end
apply_projection(paths::Vector{Matrix{Float64}}, P::Matrix{Float64}) = [p * P for p in paths]

# =====================================================================
# PART 2 - Features WITHOUT signature (baseline)
# =====================================================================

"Naive baseline: flatten the raw (time x channel) matrix into one vector."
flat_features(pts::Matrix{Float64}) = vec(permutedims(pts))

"Apply a feature function to every sample and stack into an N x F matrix."
build_matrix(paths, featfun) = permutedims(reduce(hcat, [featfun(p) for p in paths]))

# =====================================================================
# PART 3 - Generic UEA/UCR (.ts) loader
# =====================================================================

"Downloads and unzips a dataset from the UEA/UCR archive. Uses the
same URL pattern the `aeon` Python package uses internally. Your
machine needs unrestricted internet access and a working `unzip`
on PATH (both standard on macOS/Linux)."
function download_uea(name::String; dest = "./uea_data")
    mkpath(dest)
    zip_path = joinpath(dest, "$(name).zip")
    if !isfile(zip_path)
        url = "https://timeseriesclassification.com/aeon-toolkit/$(name).zip"
        println("Downloading $name from $url ...")
        download(url, zip_path)
    end
    extract_dir = joinpath(dest, name)
    if !isdir(extract_dir) || isempty(readdir(extract_dir))
        mkpath(extract_dir)
        run(`unzip -o -q $zip_path -d $extract_dir`)
    end
    return extract_dir
end

"Parses one .ts file (UEA/UCR 'sktime/aeon' format). For multivariate
series, each data line has one colon-separated block per channel,
comma-separated values within a channel, and the class label as the
final colon-separated field. Assumes equal-length series and no
missing values (true for the 4 datasets used in this tutorial)."
function read_ts_file(path::String)
    series = Vector{Matrix{Float64}}()
    labels = String[]
    in_data = false
    for line in eachline(path)
        line = strip(line)
        isempty(line) && continue
        if startswith(line, "@data")
            in_data = true
            continue
        end
        startswith(line, "@") && continue      # metadata line, skip
        !in_data && continue

        parts = split(line, ':')
        label = String(strip(parts[end]))
        channels = parts[1:end-1]
        d = length(channels)
        vals0 = [parse(Float64, s) for s in split(channels[1], ',')]
        n = length(vals0)
        M = Matrix{Float64}(undef, n, d)
        M[:, 1] = vals0
        for c in 2:d
            M[:, c] = [parse(Float64, s) for s in split(channels[c], ',')]
        end
        push!(series, M)
        push!(labels, label)
    end
    return series, labels
end

"High-level loader: downloads (if needed), extracts, and parses both
the TRAIN and TEST splits of a UEA dataset. Labels are coerced to
Int when every label string parses as an integer (true for most UEA
datasets), otherwise left as strings (e.g. BasicMotions, whose
labels are activity names)."
function load_uea(name::String; dest = "./uea_data")
    dir = download_uea(name; dest = dest)
    train_paths, ytr_str = read_ts_file(joinpath(dir, "$(name)_TRAIN.ts"))
    test_paths,  yte_str = read_ts_file(joinpath(dir, "$(name)_TEST.ts"))
    ytr = all(s -> tryparse(Int, s) !== nothing, ytr_str) ?
          parse.(Int, ytr_str) : ytr_str
    yte = all(s -> tryparse(Int, s) !== nothing, yte_str) ?
          parse.(Int, yte_str) : yte_str
    println("$name: train=$(length(train_paths))  test=$(length(test_paths))  ",
            "d=$(size(train_paths[1], 2))  len=$(size(train_paths[1], 1))  ",
            "classes=$(sort(unique(ytr)))")
    return train_paths, ytr, test_paths, yte
end

# =====================================================================
# PART 4 - Classifiers (6 total; k-NN was dropped from this battery)
# =====================================================================
#
# NOTE ON REPRODUCIBILITY: `build_forest` (Random Forest) takes an
# explicit `rng` seed and is therefore deterministic across runs.
# `build_tree` (single Decision Tree) and `build_adaboost_stumps`
# (AdaBoost) do NOT expose a seed in this version of DecisionTree.jl,
# so their accuracy can vary a few points between runs on the exact
# same data. If you need clean, reproducible numbers for a talk or
# paper, average several runs of these two classifiers (or set
# `Random.seed!(...)` right before calling them, which helps but is
# not guaranteed to be honored by every code path internally).
# All other classifiers here are fully deterministic given the data.

"Statistics-qualified: with `using Oscar` (and therefore Hecke/Nemo/
AbstractAlgebra) loaded, names like `mean`, `std`, `var` are exported
by multiple packages with different meanings (e.g. `var` = a
polynomial ring variable). Every call below is qualified with
`Statistics.` to avoid ambiguity errors."

function tree_accuracy(Xtr, ytr, Xte, yte)
    model = build_tree(ytr, Xtr)
    return Statistics.mean(apply_tree(model, Xte) .== yte)
end

function rf_accuracy(Xtr, ytr, Xte, yte; n_trees = 500, seed = 1)
    model = build_forest(ytr, Xtr, -1, n_trees, 0.7, -1; rng = seed)
    preds = apply_forest(model, Xte)
    return Statistics.mean(preds .== yte)
end

function adaboost_accuracy(Xtr, ytr, Xte, yte; n_iter = 100)
    model, coeffs = build_adaboost_stumps(ytr, Xtr, n_iter)
    preds = apply_adaboost_stumps(model, coeffs, Xte)
    return Statistics.mean(preds .== yte)
end

"Standardizes features to zero mean / unit variance, fit on TRAIN
and applied to both TRAIN and TEST (no leakage)."
function standardize(Xtr, Xte)
    mu = Statistics.mean(Xtr, dims = 1)
    sd = Statistics.std(Xtr, dims = 1)
    sd[sd .== 0] .= 1.0
    return (Xtr .- mu) ./ sd, (Xte .- mu) ./ sd
end

"Gaussian Naive Bayes, implemented directly (no extra dependency)."
function naive_bayes_accuracy(Xtr, ytr, Xte, yte)
    classes = sort(unique(ytr))
    mus = Dict(c => vec(Statistics.mean(Xtr[ytr .== c, :], dims = 1)) for c in classes)
    vars_ = Dict(c => vec(Statistics.var(Xtr[ytr .== c, :], dims = 1)) .+ 1e-6 for c in classes)
    logpriors = Dict(c => log(count(==(c), ytr) / length(ytr)) for c in classes)
    preds = similar(yte)
    for i in 1:size(Xte, 1)
        x = @view Xte[i, :]
        best_c, best_ll = classes[1], -Inf
        for c in classes
            ll = logpriors[c] - 0.5 * sum(log.(2π .* vars_[c]) .+ (x .- mus[c]).^2 ./ vars_[c])
            if ll > best_ll; best_ll = ll; best_c = c; end
        end
        preds[i] = best_c
    end
    return Statistics.mean(preds .== yte)
end

"Row-wise numerically-stable softmax."
function softmax_rows(Z)
    M = Z .- maximum(Z, dims = 2)
    E = exp.(M)
    return E ./ sum(E, dims = 2)
end

"Multinomial (softmax) logistic regression, trained with plain
full-batch gradient descent. Implemented from scratch in pure Julia
to avoid pulling in all of MLJ as a dependency (which would add a
lot of extra install surface on top of Oscar). `lr=1.0, epochs=3000`
were tuned empirically to reach convergence comparable to sklearn's
L-BFGS solver on standardized features - the earlier default
(`lr=0.1, epochs=300`) under-trains and under-reports accuracy by
several points."
function train_logreg(Xtr, ytr; lr = 1.0, epochs = 3000, lambda = 1e-3, seed = 0)
    classes = sort(unique(ytr))
    C = length(classes)
    class_idx = Dict(c => i for (i, c) in enumerate(classes))
    N, D = size(Xtr)
    Y = zeros(N, C)
    for i in 1:N; Y[i, class_idx[ytr[i]]] = 1.0; end

    Random.seed!(seed)
    W = 0.01 .* randn(D, C); b = zeros(1, C)
    for epoch in 1:epochs
        Z = Xtr * W .+ b
        P = softmax_rows(Z)
        gradZ = (P .- Y) ./ N
        gradW = Xtr' * gradZ .+ lambda .* W
        gradb = sum(gradZ, dims = 1)
        W .-= lr .* gradW
        b .-= lr .* gradb
    end
    return W, b, classes
end

function logreg_accuracy(Xtr, ytr, Xte, yte; kwargs...)
    Ztr, Zte = standardize(Xtr, Xte)
    W, b, classes = train_logreg(Ztr, ytr; kwargs...)
    P = softmax_rows(Zte * W .+ b)
    preds = [classes[argmax(P[i, :])] for i in 1:size(P, 1)]
    return Statistics.mean(preds .== yte)
end

"SVM with an RBF kernel, via LIBSVM.jl. Note LIBSVM expects data as
d x N (features x samples), the transpose of our usual N x d layout -
hence the `permutedims` calls."
function svm_accuracy(Xtr, ytr, Xte, yte)
    Ztr, Zte = standardize(Xtr, Xte)
    model = svmtrain(permutedims(Ztr), ytr; kernel = LIBSVM.Kernel.RadialBasis)
    preds, _ = svmpredict(model, permutedims(Zte))
    return Statistics.mean(preds .== yte)
end

CLASSIFIERS_ALL = [
    ("DecisionTree (single)", tree_accuracy),
    ("RandomForest",          rf_accuracy),
    ("AdaBoost (stumps)",     adaboost_accuracy),
    ("Naive Bayes (Gauss.)",  naive_bayes_accuracy),
    ("Logistic Regression",   logreg_accuracy),
    ("SVM (RBF)",             svm_accuracy),
]

"Runs every classifier in CLASSIFIERS_ALL on one (Xtr,ytr,Xte,yte)
feature set, prints progress with per-classifier wall-clock time,
and appends (feature_set_name, classifier_name, num_features,
accuracy) to `results` for the final summary table. Wraps each
classifier call in try/catch so one failure doesn't abort the run."
function run_battery(name::String, Xtr, ytr, Xte, yte, results)
    println("\n=== $name  (d=$(size(Xtr,2))) ===")
    for (cname, fn) in CLASSIFIERS_ALL
        t0 = time()
        acc = try
            fn(Xtr, ytr, Xte, yte)
        catch e
            println("  $(rpad(cname,22)) FAILED: $e")
            continue
        end
        dt = time() - t0
        push!(results, (name, cname, size(Xtr,2), acc))
        println("  $(rpad(cname,22)) acc=$(round(100acc,digits=2))%  ($(round(dt,digits=1))s)")
    end
end

# =====================================================================
# PART 5 - Load all 4 datasets and run the full battery
# =====================================================================

results_multi = Tuple{String,String,Int,Float64}[]

# --- BasicMotions: d=6, len=100, 4 classes, train=40, test=40 ---
train_bm, ytr_bm, test_bm, yte_bm = load_uea("BasicMotions")
Xtr_flat_bm = build_matrix(train_bm, flat_features)
Xte_flat_bm = build_matrix(test_bm,  flat_features)
Xtr_sig_bm  = build_signature_matrix_general(train_bm, 3; augment=true)
Xte_sig_bm  = build_signature_matrix_general(test_bm,  3; augment=true)
Xtr_sig_bm, Xte_sig_bm = drop_constant_columns(Xtr_sig_bm, Xte_sig_bm)

run_battery("BasicMotions raw-flatten", Xtr_flat_bm, ytr_bm, Xte_flat_bm, yte_bm, results_multi)
run_battery("BasicMotions sig depth3",  Xtr_sig_bm,  ytr_bm, Xte_sig_bm,  yte_bm, results_multi)

# --- Cricket: d=6, len=1197 (long series!), 12 classes, train=108, test=72 ---
train_cr, ytr_cr, test_cr, yte_cr = load_uea("Cricket")
Xtr_flat_cr = build_matrix(train_cr, flat_features)
Xte_flat_cr = build_matrix(test_cr,  flat_features)
Xtr_sig_cr  = build_signature_matrix_general(train_cr, 3; augment=true)
Xte_sig_cr  = build_signature_matrix_general(test_cr,  3; augment=true)
Xtr_sig_cr, Xte_sig_cr = drop_constant_columns(Xtr_sig_cr, Xte_sig_cr)

run_battery("Cricket raw-flatten", Xtr_flat_cr, ytr_cr, Xte_flat_cr, yte_cr, results_multi)
run_battery("Cricket sig depth3",  Xtr_sig_cr,  ytr_cr, Xte_sig_cr,  yte_cr, results_multi)

# --- ArticularyWordRecognition: d=9, len=144, 25 classes, train=275, test=300 ---
train_awr, ytr_awr, test_awr, yte_awr = load_uea("ArticularyWordRecognition")
Xtr_flat_awr = build_matrix(train_awr, flat_features)
Xte_flat_awr = build_matrix(test_awr,  flat_features)
Xtr_sig_awr  = build_signature_matrix_general(train_awr, 3; augment=true)
Xte_sig_awr  = build_signature_matrix_general(test_awr,  3; augment=true)
Xtr_sig_awr, Xte_sig_awr = drop_constant_columns(Xtr_sig_awr, Xte_sig_awr)

run_battery("AWR raw-flatten", Xtr_flat_awr, ytr_awr, Xte_flat_awr, yte_awr, results_multi)
run_battery("AWR sig depth3",  Xtr_sig_awr,  ytr_awr, Xte_sig_awr,  yte_awr, results_multi)

# --- NATOPS: d=24, len=51, 6 classes, train=180, test=180 ---
# We deliberately use the FULL d=24 channels (no random projection)
# for both signature depths below, and instead trade off depth k:
#   k=2 -> 648 features   (cheap)
#   k=3 -> 16272 features (still tractable with only 180 samples,
#          and empirically the best trade-off we found - projecting
#          channels down to d'=5 or d'=8 before the signature scored
#          WORSE than raw-flatten; keeping d=24 and lowering k instead
#          beat raw-flatten cleanly). k=4 (~407k features) gave a
#          negligible further gain (+0.6pp) for ~25x more features
#          and roughly a minute of extra compute - included here only
#          as a lesson in diminishing returns, not because it's the
#          recommended setting.
train_nat, ytr_nat, test_nat, yte_nat = load_uea("NATOPS")
Xtr_flat_nat = build_matrix(train_nat, flat_features)
Xte_flat_nat = build_matrix(test_nat,  flat_features)
Xtr_sig_nat  = build_signature_matrix_general(train_nat, 2; augment=true)
Xte_sig_nat  = build_signature_matrix_general(test_nat,  2; augment=true)
Xtr_sig_nat, Xte_sig_nat = drop_constant_columns(Xtr_sig_nat, Xte_sig_nat)

run_battery("NATOPS raw-flatten",     Xtr_flat_nat, ytr_nat, Xte_flat_nat, yte_nat, results_multi)
run_battery("NATOPS sig depth2(d24)", Xtr_sig_nat,  ytr_nat, Xte_sig_nat,  yte_nat, results_multi)

# k=3, full d=24 channels -> 16272 features. Noticeably slower,
# especially SVM and Logistic Regression, but should not hang -
# there are still only 180 training samples.
Xtr_sig_nat3 = build_signature_matrix_general(train_nat, 3; augment=true)
Xte_sig_nat3 = build_signature_matrix_general(test_nat,  3; augment=true)
Xtr_sig_nat3, Xte_sig_nat3 = drop_constant_columns(Xtr_sig_nat3, Xte_sig_nat3)

run_battery("NATOPS sig depth3(d24)", Xtr_sig_nat3, ytr_nat, Xte_sig_nat3, yte_nat, results_multi)

# =====================================================================
# PART 6 - Final summary table
# =====================================================================

println("\n", "="^75)
println(rpad("dataset/features", 26), rpad("classifier", 24), rpad("#feat", 7), "acc")
println("="^75)
for (fname, cname, nf, acc) in results_multi
    println(rpad(fname, 26), rpad(cname, 24), rpad(string(nf), 7),
            "$(round(100acc, digits=2))%")
end
println("="^75)
