struct DiagonalOperator{T<:Number} <: AbstractLinearOperator{T,T}
    n::Integer
    θ::Optional{AbstractVector{T}}
    id::Any
end

DiagonalOperator{T}(n::Integer) where{T} =
    DiagonalOperator{T}(
        n,
        nothing,
        uuid4(GLOBAL_RNG)
    )

Domain(A::DiagonalOperator) = A.n
Range(A::DiagonalOperator) = A.n
param(A::DiagonalOperator) = [A.θ]
nparam(A::DiagonalOperator) = 1

function init(A::DiagonalOperator{T}, pv::Optional{ParameterVector}) where {T}
    θ = rand(T, A.n)
    if !isnothing(pv)
        pv[A.id] = [θ]
    end
    return [θ]
end

adjoint(A::DiagonalOperator) = A

id(A::DiagonalOperator) = A.id
*(A::DiagonalOperator{T}, x::AbstractVector{T}) where {T} = A.θ.*x
*(A::DiagonalOperator{T}, x::AbstractVecOrMat{T}) where {T} = A.θ.*x

(A::DiagonalOperator{T})(θ::AbstractVector{T}) where {T} =
    DiagonalOperator{T}(
        A.n,
        θ,
        A.id
    )

(A::DiagonalOperator{T})(pv::ParameterVector) where {T} =
    DiagonalOperator{T}(
        A.n,
        pv[A.id][1],
        A.id
    )
