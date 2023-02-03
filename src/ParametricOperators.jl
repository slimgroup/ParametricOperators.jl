module ParametricOperators

import Base: +, -, *, /, ^, ∘
import Base: adjoint, kron

using Base.Broadcast: BroadcastFunction
using ChainRulesCore
using DataStructures: OrderedDict
using FFTW

# ==== Includes ====

# Common types and functions
include("ParCommon.jl")

# Base operator definition and functionality
include("ParOperator.jl")

# Operator wrappers
include("ParAdjoint.jl")
include("ParParameterized.jl")

# Operator combinations
include("ParCompose.jl")
include("ParKron.jl")

# Operator definitions
include("ParMatrix.jl")
include("ParDFT.jl")

end