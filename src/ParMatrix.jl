export ParMatrix

"""
Dense matrix operator.
"""
struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,External}
    m::Int
    n::Int
    ParMatrix(T, m, n) = new{T}(m, n)
    ParMatrix(m, n) = new{Float64}(m, n)
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m

complexity(A::ParMatrix{T}) where {T} = elementwise_multiplication_cost(T)*A.n*A.m

function init!(A::ParMatrix{T}, d::Parameters) where {T}
    d[A] = rand(T, A.m, A.n)/sqrt(A.m*A.n)
end

(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParMatrix{T},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractVector{T}} = A.params'*x
(A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V})(x::X) where {T,V,X<:AbstractMatrix{T}} = A.params'*x
*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractVector{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParMatrix{T},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractVector{T}} = x*A.params'
*(x::X, A::ParParameterized{T,T,Linear,ParAdjoint{T,T,Parametric,ParMatrix{T}},V}) where {T,V,X<:AbstractMatrix{T}} = x*A.params'