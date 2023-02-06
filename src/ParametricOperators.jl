module ParametricOperators

import Base: +, -, *, /, ^, ∘
import Base: adjoint, kron

using Base.Broadcast: BroadcastFunction
using ChainRulesCore
using Combinatorics
using DataStructures: OrderedDict, DefaultDict
using FFTW
using LaTeXStrings

# ==== Includes ====

# Common types and functions
include("ParCommon.jl")

# Base operator definition and functionality
include("ParOperator.jl")
include("ParOperatorViz.jl")

# Tree optimization
include("MachineModel.jl")
include("ASTOptimization.jl")

# Operator wrappers
include("ParAdjoint.jl")
include("ParParameterized.jl")

# Operator combinations
include("ParIdentity.jl") # Include above for use in transforms, etc.
include("ParCompose.jl")
include("ParKron.jl")

# Operator definitions
include("ParMatrix.jl")
include("ParDFT.jl")

end