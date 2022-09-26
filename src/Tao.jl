module Tao

import Base: +, -, *, /, ∘
import Base: adjoint, kron

using ChainRulesCore
using FFTW: fft, ifft
using Match
using Random: GLOBAL_RNG
using UUIDs: uuid4, UUID

include("TaoCommon.jl")

include("TaoOperator.jl")
include("TaoOperatorTraits.jl")

include("TaoAdd.jl")
include("TaoCompose.jl")
include("TaoKron.jl")

include("TaoConvert.jl")
include("TaoIdentity.jl")
include("TaoMatrix.jl")
include("TaoDiagonal.jl")
include("TaoFunction.jl")
include("TaoBias.jl")
include("TaoRestriction.jl")
include("TaoDFT.jl")
include("TaoDRFT.jl")

end # module Tao
