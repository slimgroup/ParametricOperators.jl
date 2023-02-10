module ParametricOperators

import Base: +, -, *, /, ^, ∘
import Base: adjoint, kron

using Base.Broadcast: BroadcastFunction
using ChainRulesCore
using Combinatorics
using CUDA
using DataStructures: OrderedDict, DefaultDict
using FFTW
using LaTeXStrings
using Match
using MPI

# ==== Includes ====

# Common types and functions
include("ParCommon.jl")

# Base operator definition and functionality
include("ParOperator.jl")
include("ParOperatorViz.jl")

# Tree optimization
include("MachineModel.jl")
include("ASTOptimization.jl")

# Operator distribution
include("ParDistributed.jl")
include("ParBroadcasted.jl")
include("ParRepartition.jl")

# Operator wrappers
include("ParIdentity.jl") # Include above for use in transforms, etc.
include("ParAdjoint.jl")
include("ParParameterized.jl")

# Operator combinations
include("ParCompose.jl")
include("ParKron.jl")

# Operator definitions
include("ParMatrix.jl")
include("ParDiagonal.jl")
include("ParDFT.jl")
include("ParRestriction.jl")

end