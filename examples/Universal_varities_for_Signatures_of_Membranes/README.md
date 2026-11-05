This is the supplementary computational material for the paper
"Universal Varieties for Signatures of Membranes"
(Amendola, Riffo, Schmitz).

Below is what each file is for.

Chen_weak.jl
  Builds the "weak Chen chain" and computes Jacobian ranks.
  -> Theorem 2.12 (Weak Chen's identity).

Conterexample_poly_mem.jl
  Computes the varieties P (moment/polynomial), L (axis/piecewise bilinear)
  and J = P ∩ L for (d,m,n)=(2,1,2), k=3, and shows they do not coincide.
  -> Example 4.17.

G23.jl
  Builds the ideal G_{2,3} (explicit generators) and checks the two-parameter
  Witt formula and the relation from Example 3.4.
  -> Example 4.1, Example 3.4.

U23.jl
  Builds the universal variety U_{2,3} (eliminates the lower-weight
  variables) and computes its dimension, degree and minimal generators.
  -> Example 4.3.

shuffle_verf.jl
  Numerically verifies two equivalent definitions of the two-parameter
  shuffle product (with and without inverse).
  -> Definition 2.6 / Theorem 2.8 (consistency of the shuffle product).

dimension_membrane.m2
  Macaulay2 script that computes the dimension of the image of the
  parametrization via NumericalImplicitization, for (m,n,d)=(2,4,5).
  -> Section 5 / Example 5.9 (cross-check with another computer algebra system).

dim_master.jl
  Computes dim P_{d,k,(m,n)} as the generic rank of the Jacobian of the
  parametrization A -> sigma^(k)(A * Dict), for Dict = axis or mono.
  -> Table 1 (Example 5.9, dict=axis, k=2) and Table 2 (Example 6.7, dict=mono, k=3).

degree_grob.jl
  Computes dimension, degree and number of generators exactly (via ideal
  kernel/elimination), not just a numerical rank.
  -> Degrees cited in Example 4.1, Example 4.3 and Example 4.17.

table_3_examples.jl
  Same as degree_grob.jl (exact dimension + degree + generators) but
  generalized to run full tables over d, k, m, n and type (axis/mono).
  -> Table 2 (Example 6.7), exact/with-generators version.
