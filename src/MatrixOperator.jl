import Base.*

"""
Dense matrix operator.
"""
struct MatrixOperator{T} <: AbstractLinearOperator{T,T}
    θ::Optional{Matrix{T}}
    m::Int64
    n::Int64
    id::Any
end

"""
Dense matrix constructor.
"""
MatrixOperator{T}(m::Int64, n::Int64) where {T} =
    MatrixOperator{T}(
        nothing,
        m,
        n,
        uid()
    )

# Dense matrix operator impls
Domain(A::MatrixOperator) = A.n
Range(A::MatrixOperator) = A.m
params(A::MatrixOperator) = [A.θ]
nparams(A::MatrixOperator) = 1
id(A::MatrixOperator) = A.id

function init(A::MatrixOperator{T}, pc::Optional{ParameterContainer} = nothing) where {T}
    θ = T(1)/T(A.m*A.n)*rand(T, A.m, A.n)
    if !isnothing(pc)
        pc[A.id] = θ
    end
    return [θ]
end

*(A::MatrixOperator{T}, x::V) where {T,V<:AbstractVector{T}} = A.θ*x
*(A::MatrixOperator{T}, x::M) where {T,M<:AbstractVecOrMat{T}} = A.θ*x
*(A::LinearOperatorAdjoint{T,T,MatrixOperator{T}}, x::V) where {T,V<:AbstractVector{T}} = A.inner.θ'*x
*(A::LinearOperatorAdjoint{T,T,MatrixOperator{T}}, x::M) where {T,M<:AbstractVecOrMat{T}} = A.inner.θ'*x

(A::MatrixOperator{T})(M::Matrix{T}) where {T} =
    MatrixOperator{T}(
        M,
        A.m,
        A.n,
        A.id
    )

(A::MatrixOperator{T})(pc::ParameterContainer) where {T} =
    MatrixOperator{T}(
        pc[A.id][1],
        A.m,
        A.n,
        A.id
    )