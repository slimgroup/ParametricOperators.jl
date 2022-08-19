"""
Base operator type. Represents an arbitrary map F: Dⁿ -> Rᵐ.
"""
abstract type AbstractOperator{D,R} end

"""
Datatype of the domain vector space D of operator F.
"""
ddt(::AbstractOperator{D,R}) where {D,R} = D

"""
Datatype of the range vector space R of operator F.
"""
rdt(::AbstractOperator{D,R}) where {D,R} = R

"""
Dimensionality of domain vector space D of operator F.
"""
Domain(F::AbstractOperator) = throw(TaoException("Domain() is not implemented for $(typeof(F))"))

"""
Dimensionality of range vector space R of operator F.
"""
Range(F::AbstractOperator) = throw(TaoException("Range() is not implemented for $(typeof(F))"))

"""
Parameters associated with operator F. For n-ary operators (e.g. addition),
this should return a flattened vector with the parameters of the operators
in read order.
"""
params(F::AbstractOperator) = throw(TaoException("param() is not implemented for $(typeof(F))"))

"""
Number of parameters associated with operator F.
"""
nparams(F::AbstractOperator) = throw(TaoException("nparam() is not implemented for $(typeof(F))"))

"""
Initializes operator and optionally stores the initialized value in a parameter container.
"""
init(F::AbstractOperator, ::Optional{ParameterContainer} = nothing) =
    throw(TaoException("init() is not implemented for $(typeof(F))"))

"""
ID of the given operator.
"""
id(F::AbstractOperator) = throw(TaoException("id() is not implemented for $(typeof(F))"))

"""
Recreates the operator with a new domain type.
"""
retype_domain(F::AbstractOperator{D,R}, ::T) where {D,R,T} =
    throw(TaoException("retype_domain() is not implemented for $(typeof(F))"))

"""
Calls the operator F with vector x and parameters θ.
"""
(F::AbstractOperator{D,R})(::V, ::Vararg{<:AbstractArray}) where {D,R,V<:AbstractVector{D}} =
    throw(TaoException("call() is not implemented for $(typeof(F))"))

"""
Calls the operator F with vector x and parameters θ.
"""
(F::AbstractOperator{D,R})(::V, ::ParameterContainer) where {D,R,V<:AbstractVector{D}} =
    throw(TaoException("call() is not implemented for $(typeof(F))"))

"""
Calls the operator F with multivector x and parameters θ. Concrete types should
specify this function if possible (e.g. for a matrix A and multivector V, the
impl is A*V).
"""
function (F::AbstractOperator{D,R})(x::V, θs::Vararg{<:AbstractArray}) where {D,R,V<:AbstractVecOrMat{D}}
    return mapreduce(v -> F(v, θs...), hcat, eachcol(x))
end