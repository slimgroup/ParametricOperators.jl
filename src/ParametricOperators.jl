module ParametricOperators

import Base: +, -, *, /, ∘
import Base: adjoint, kron, size, zero, view
import Base: getindex, setindex!, IndexStyle, IndexLinear

using ChainRulesCore
using DataStructures: OrderedDict
using FFTW
using Random: GLOBAL_RNG
using UUIDs: uuid4, UUID

include("ParCommon.jl")
include("MultiTypeVector.jl")

include("ParOperator.jl")
include("ParOperatorTraits.jl")

include("ParAdjoint.jl")
include("ParParameterized.jl")

include("ParAdd.jl")
include("ParCompose.jl")
include("ParKron.jl")

include("ParMatrix.jl")
include("ParDFT.jl")

end