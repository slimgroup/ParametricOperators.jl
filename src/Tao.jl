module Tao

    import Base.+, Base.-, Base.*, Base./, Base.∘
    import Base.adjoint, Base.kron
    import Random.GLOBAL_RNG
    import UUIDs.uuid4

    using FFTW
    using Match

    include("Types.jl")
    include("Utils.jl")

    export TaoException, Optional
    export uid, subtype, print0, println0

    include("Operator.jl")
    include("AddOperator.jl")
    include("MulOperator.jl")
    include("KronOperator.jl")
    include("PromoteOperator.jl")
    include("CompositionOperator.jl")
    include("FunctionOperator.jl")

    export Linearity, Linear, NonLinear
    export Parametricity, Parametric, NonParametric
    export ParamOrdering, LeftFirst, RightFirst
    export Operator, Parameterized, Adjoint
    export DDT, RDT, Domain, Range, nparams, init, id
    
    export AddOperator
    export MulOperator
    export KronOperator, ⊗
    export PromoteOperator
    export CompositionOperator
    export FunctionOperator

    include("MatrixOperator.jl")
    include("DiagonalOperator.jl")
    include("DFTOperator.jl")
    include("DRFTOperator.jl")
    include("RestrictionOperator.jl")

    export MatrixOperator
    export DiagonalOperator
    export DFTOperator
    export DRFTOperator
    export RestrictionOperator
    
end