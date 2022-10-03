module ParametricOperators

import Base: +, -, *, /, ∘
import Base: adjoint, kron

using CUDA
using ChainRulesCore
using FFTW
using LinearAlgebra
using Random: GLOBAL_RNG
using UUIDs: UUID, uuid4

include("ParCommon.jl")
include("ParOperator.jl")
include("ParOperatorTraits.jl")
include("ParMatrix.jl")
include("ParDiagonal.jl")
include("ParRestriction.jl")
include("ParDFT.jl")

include("ParAdd.jl")
include("ParCompose.jl")
include("ParKron.jl")

end