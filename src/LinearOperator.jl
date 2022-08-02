"""
Abstract linear operator type. Represents a linear map A: Dⁿ → Rᵐ.
"""
abstract type AbstractLinearOperator{D, R} end

"""
Base linear operator exception type.
"""
struct LinearOperatorException <: Exception
    msg::String
end

"""
Gets the size of the Domain vector space of A.
"""
Domain(A::AbstractLinearOperator) = throw(LinearOperatorException("Domain() is not implemented for $(typeof(A))"))

"""
Gets the size of the Range vector space of A.
"""
Range(A::AbstractLinearOperator) = throw(LinearOperatorException("Range() is not implemented for $(typeof(A))"))

"""
Gets the parameters of A as a vector.
"""
param(A::AbstractLinearOperator) = throw(LinearOperatorException("param() is not implemented for $(typeof(A))"))

"""
Gets the number of parameters of A.
"""
nparam(A::AbstractLinearOperator) = throw(LinearOperatorException("nparam() is not implemented for $(typeof(A))"))

"""
Initializes A and returns a vector of its parameters. If a ParameterVector is
passed, the parameters of a A are also stored inside with a key equal to the id
of A.
"""
init(A::AbstractLinearOperator, ::Optional{ParameterVector} = nothing) =
    throw(LinearOperatorException("init() is not implemented for $(typeof(A))"))

"""
Returns a new operator representing the adjoint of A.
"""
adjoint(A::AbstractLinearOperator) = throw(LinearOperatorException("adjoint() is not implemented for $(typeof(A))"))

"""
Gets the unique ID of A.
"""
id(A::AbstractLinearOperator) = throw(LinearOperatorException("id() is not implemented for $(typeof(A))"))

"""
Applies A to a vector x.
"""
*(A::AbstractLinearOperator{D,R}, x::AbstractVector{D}) where {D,R} =
    throw(LinearOperatorException("mvp is not implemented for $(typeof(A))"))

"""
Applies A to a multivector x.
"""
*(A::AbstractLinearOperator{D,R}, x::AbstractVecOrMat{D}) where {D,R} =
    throw(LinearOperatorException("mmp is not implemented for $(typeof(A))"))

"""
Parameterizes A using the passed parameters θs.
"""
(A::AbstractLinearOperator)(θs::Any...) = throw(LinearOperatorException("Cannot parameterize operator of type $(typeof(A))"))

"""
Parameterizes A using the parameter vector pv.
"""
(A::AbstractLinearOperator)(pv::ParameterVector) =
    throw(LinearOperatorException("Cannot parameterize operator of type $(typeof(A)) with parameter vector"))

"""
Gets the Domain datatype of A.
"""
ddt(::AbstractLinearOperator{D,R}) where{D,R} = D

"""
Gets the Range datatype of A.
"""
rdt(::AbstractLinearOperator{D,R}) where{D,R} = R
