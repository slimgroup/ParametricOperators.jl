module Tao

    # Common package types and utilities
    include("Types.jl")
    include("Utils.jl")
    include("ParameterContainer.jl")

    export TaoException, Optional
    export ParameterContainer

    # Abstract operator types and functions
    include("Operator.jl")
    include("LinearOperator.jl")

    export AbstractOperator, AbstractLinearOperator
    export ddt, rdt, Domain, Range, param, nparam, init, id

    # Operator combination types and functions
    include("AddOperator.jl")
    include("ElementwiseOperator.jl")
    include("CompositionOperator.jl")

    export AddOperator, AddLinearOperator
    export ElementwiseOperator, CompositionOperator

    # Specific operator types and functions
    include("MatrixOperator.jl")
    include("BiasOperator.jl")
    include("FunctionOperator.jl")

    export MatrixOperator, BiasOperator, FunctionOperator

end