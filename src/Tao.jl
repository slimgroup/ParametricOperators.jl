module Tao

using FFTW, LinearAlgebra

import Base.*, Base.+
import LinearAlgebra.adjoint

import Random.GLOBAL_RNG
import UUIDs.uuid4

export AbstractLinearOperator, MatrixOperator, DiagonalOperator, DFTOperator, DRFTOperator, ⊗, ParameterVector
export IdentityOperator, RestrictionOperator, ddt, rdt, Domain, Range, param, nparam, init, count_params
export RepartitionOperator, print_seq, println0

include("Types.jl")
include("LinearOperator.jl")
include("AddOperator.jl")
include("MulOperator.jl")
include("KronOperator.jl")
include("MatrixOperator.jl")
include("DiagonalOperator.jl")
include("RestrictionOperator.jl")
include("IdentityOperator.jl")
include("DFTOperator.jl")
include("DRFTOperator.jl")
include("Utils.jl")
include("RepartitionOperator.jl")

end
