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
nparams(A::ParDiagonal) = 1
init(A::ParDiagonal{T}) where {T} = [rand(T, A.n)]

(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params[1].*x
(A::ParParameterized{T,T,Linear,ParDiagonal{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params[1].*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = conj(A.params[1]).*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = conj(A.params[1]).*x
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractVector{T}} = x.*A.params[1]
*(x::X, A::ParParameterized{T,T,Linear,ParDiagonal{T},V}) where {T,V,X<:AbstractMatrix{T}} = x.*A.params[1]
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractVector{T}} = x.*conj(A.params[1])
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParDiagonal{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x.*conj(A.params[1])
