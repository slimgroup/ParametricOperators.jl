module ParametricOperators

greet() = print("Hello World!")

# ==== Imports ====

import Base: +, -, *, /, ∘
import Base: adjoint, kron

using Base: Iterators
using ChainRulesCore
using CUDA
using DataStructures
using FFTW: fft, ifft
using LRUCache
using MPI
using Random: GLOBAL_RNG

if CUDA.functional()
    @info "ParametricOperators.jl successfully loaded CUDA.jl. Defining operator implementations for CUDA types."
end

# ==== Includes ====

# Common types and functions
include("ParCommon.jl")

# Base operator definition, functionality, and derivative rules
include("ParOperator.jl")
include("ParOperatorRules.jl")

# Operator distribution
include("ParDistributed.jl")
include("ParBroadcasted.jl")
include("ParRepartition.jl")

# Operator wrappers
include("ParAdjoint.jl")
include("ParParameterized.jl")

# Operator combinations
include("ParIdentity.jl") # Include above for use in other files
include("ParAdd.jl")
include("ParCompose.jl")
include("ParKron.jl")

# Operator definitions
include("ParMatrix.jl")
include("ParDiagonal.jl")
include("ParBias.jl")
include("ParDFT.jl")
include("ParRestriction.jl")
include("ParFunction.jl")

end
