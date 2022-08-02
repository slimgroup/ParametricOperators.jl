import Base.*
using ChainRulesCore

struct MatrixOperator{T<:Number} <: AbstractLinearOperator{T,T}
    m::Integer
    n::Integer
    θ::Optional{AbstractMatrix{T}}
    adjoint::Bool
    id::Any
end

MatrixOperator{T}(m::Integer, n::Integer) where{T} =
    MatrixOperator{T}(
        m,
        n,
        nothing,
        false,
        uuid4(GLOBAL_RNG)
    )

Domain(A::MatrixOperator) = A.adjoint ? A.m : A.n
Range(A::MatrixOperator) = A.adjoint ? A.m : A.m
param(A::MatrixOperator) = [A.θ]
nparam(A::MatrixOperator) = 1

function init(A::MatrixOperator{T}, pv::Optional{ParameterVector}) where {T}
    θ = rand(T, A.m, A.n)
    if !isnothing(pv)
        pv[A.id] = [θ]
    end
    return [θ]
end

adjoint(A::MatrixOperator{T}) where {T} =
    MatrixOperator{T}(
        A.m,
        A.n,
        A.θ,
        !A.adjoint,
        A.id
    )

id(A::MatrixOperator) = A.id
*(A::MatrixOperator{T}, x::AbstractVector{T}) where {T} = A.adjoint ? A.θ'*x : A.θ*x
*(A::MatrixOperator{T}, x::AbstractVecOrMat{T}) where {T} = A.adjoint ? A.θ'*x : A.θ*x

(A::MatrixOperator{T})(θ::AbstractMatrix{T}) where {T} =
    MatrixOperator{T}(
        A.m,
        A.n,
        θ,
        A.adjoint,
        A.id
    )

(A::MatrixOperator{T})(pv::ParameterVector) where {T} =
    MatrixOperator{T}(
        A.m,
        A.n,
        pv[A.id][1],
        A.adjoint,
        A.id
    )

function ChainRulesCore.rrule(::typeof(*), A::MatrixOperator{T}, x::AbstractVector{T}) where {T}
    y = A*x
    function op_matrix_pullback(∇y)
        ∇f = NoTangent()
        ∇A = Tangent{MatrixOperator{T}}(θ = @thunk(A.adjoint ? x*∇y' : ∇y*x'))
        ∇x = @thunk(A.adjoint ? A.θ * ∇y : A.θ'*∇y)
        return ∇f, ∇A, ∇x
    end
    return y, op_matrix_pullback
end
