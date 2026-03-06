"""
Welcome to SignatureTensors v0.1

This package allows computing and working with signatures of rough paths
using algebraic methods within OSCAR.

For more information, visit:
https://github.com/leonardSchmitz/signature-tensors-in-OSCAR
"""
module SignatureTensors
using Oscar;
using BenchmarkTools;



#greet() = print("Hello World!")

include("typesetting.jl")
include("LieOperations.jl")
include("tensorOperations.jl")
include("FreeTruncSigAlgMultiv.jl")
include("matrixOperations.jl")
include("TruncatedTensorAlgebra.jl")
include("tensorLearningSch25.jl")
include("Benchmarkfunction.jl")

end 
