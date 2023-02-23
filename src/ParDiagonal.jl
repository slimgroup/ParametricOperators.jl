export ParDiagonal

"""
Diagonal matrix (elementwise) operator.
"""
struct ParDiagonal{T} <: ParLinearOperator{T,T,Parametric,External}
    n::Int
    ParDiagonal(T, n) = new{T}(n)
    ParDiagonal(n) = new{Float64}(n)
end

Domain(A::ParDiagonal) = A.n
Range(A::ParDiagonal) = A.n

complexity(A::ParDiagonal{T}) where {T} = elementwise_multiplication_cost(T)*A.n

function init!(A::ParDiagonal{T}, d::Parameters) where {T}
    d[A] = rand(T, A.n)
end

(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params.*x
(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params.*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = conj(A.params[A.op.op]).*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = conj(A.params[A.op.op]).*x
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractVector{T}} = x.*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractMatrix{T}} = x.*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractVector{T}} = x.*conj(A.params[A.op.op])
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x.*conj(A.params[A.op.op])