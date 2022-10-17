module ParametricOperators

import Base: +, -, *, /, ∘
import Base: adjoint,  kron
import Base: get, replace!, getindex, setindex!, push!, append!, size, broadcasted, view
import Base: IndexStyle, IndexLinear
import Base.Iterators

using ChainRulesCore
using CUDA
using DataStructures: OrderedDict
using FFTW
using Random: GLOBAL_RNG
using UUIDs: UUID, uuid4

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
include("ParBias.jl")
include("ParIdentity.jl")
include("ParDFT.jl")
include("ParFunction.jl")

end