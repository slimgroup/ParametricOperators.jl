module ParametricOperators

greet() = print("Hello World!")

# ==== Imports ====

import Base: +, -, *, /, ∘
import Base: adjoint, kron

using Base: Iterators
using ChainRulesCore
using DataStructures
using FFTW: fft, ifft
using MPI
using Random: GLOBAL_RNG

# ==== Includes ====

# Common types and functions
include("ParCommon.jl")

# Base operator definition and functionality
include("ParOperator.jl")

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
include("ParDFT.jl")
include("ParRestriction.jl")
# include("ParFunction.jl")

end
