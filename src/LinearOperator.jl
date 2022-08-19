import Base.*
import Base.adjoint

"""
Base linear operator type. Represents a linear map A: Dⁿ -> Rᵐ.
"""
abstract type AbstractLinearOperator{D,R} <: AbstractOperator{D,R} end

"""
Wrapper type around linear operator for taking adjoints.
"""
struct LinearOperatorAdjoint{D,R,A<:AbstractLinearOperator{D,R}} <: AbstractLinearOperator{R,D}
    inner::A
end

# Operator interface impls
Domain(L::LinearOperatorAdjoint{D,R,A}) where {D,R,A<:AbstractLinearOperator{D,R}} = Range(L.inner)
Range(L::LinearOperatorAdjoint{D,R,A}) where {D,R,A<:AbstractLinearOperator{D,R}} = Domain(L.inner)
params(L::LinearOperatorAdjoint{D,R,A}) where {D,R,A<:AbstractLinearOperator{D,R}} = param(L.inner)
nparams(L::LinearOperatorAdjoint{D,R,A}) where {D,R,A<:AbstractLinearOperator{D,R}} = nparam(L.inner)
init(L::LinearOperatorAdjoint{D,R,A}, pc::Optional{ParameterContainer} = nothing) where {D,R,A<:AbstractLinearOperator{D,R}} = init(L.inner, pc)

"""
Adjoint of any given linear operator, returning a LinearOperatorAdjoint Wrapper
struct.
"""
adjoint(A::AbstractLinearOperator{D,R}) where {D,R} = LinearOperatorAdjoint{D,R,typeof(A)}(A)

"""
Multiplication of linear operator with vector x.
"""
*(A::AbstractLinearOperator{D,R}, ::V) where {D,R,V<:AbstractVector{D}} =
    throw(TaoException("mvp() is not implemented for $(typeof(A))"))

"""
Multiplication of linear operator with multivector x.
"""
*(A::AbstractLinearOperator{D,R}, x::M) where {D,R,M<:AbstractVecOrMat{D}} =
    mapreduce(v -> A*v, hcat,eachcol(x))

"""
Parameterization of linear operator with parameters θ.
"""
(A::AbstractLinearOperator)(::Vararg{<:AbstractArray}) = throw(TaoException("parameterization is not implemented for $(typeof(A))"))

"""
Parameterization of linear operator with parameters θ from parameter container.
"""
(A::AbstractLinearOperator)(::ParameterContainer) = throw(TaoException("parameterization is not implemented for $(typeof(A))"))