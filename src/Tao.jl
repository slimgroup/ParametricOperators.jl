module Tao

    # Common package types and utilities
    include("Types.jl")
    include("Utils.jl")
    include("ParameterContainer.jl")

    export issubsettypeof
    export TaoException, Optional
    export ParameterContainer

    # Abstract operator types and functions
    include("Operator.jl")
    include("LinearOperator.jl")

    export AbstractOperator, AbstractLinearOperator
    export ddt, rdt, Domain, Range, param, nparam, init, id

    # Operator combination types and functions
    include("ElementwiseOperator.jl")
    include("CompositionOperator.jl")
    include("AddOperator.jl")
    include("MulOperator.jl")
    include("KronOperator.jl")

    export ElementwiseOperator, CompositionOperator
    export AddOperator, AddLinearOperator, MulOperator, KronOperator
    export ⊗

    # Specific operator types and functions
    include("DFTOperator.jl")
    include("DRFTOperator.jl")
    include("MatrixOperator.jl")
    include("BiasOperator.jl")
    include("FunctionOperator.jl")

    export MatrixOperator, BiasOperator, FunctionOperator, DFTOperator, DRFTOperator

end