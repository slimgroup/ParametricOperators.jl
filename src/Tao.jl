module Tao

    include("Types.jl")
    include("Utils.jl")
    include("ParameterVector.jl")
    include("Operator.jl")
    include("LinearOperator.jl")

    export Optional, ParameterVector
    export new_id
    export AbstractOperator, ElementwiseOperator
    export ElementwiseOperator, CompositionOperator
    export AbstractLinearOperator, LinearOperatorException, LinearOperatorAdjoint
    export Domain, Range, ddt, rdt, id, init, param, nparam

    include("BiasOperator.jl")
    include("MatrixOperator.jl")
    
    export BiasOperator
    export MatrixOperator

    include("AddOperator.jl")

    export AddOperator

end