import Base.*

struct MatrixOperator{T} <: AbstractLinearOperator{T,T}
    m::Int64
    n::Int64
    θ::Optional{Matrix{T}}
    id::Any
end

MatrixOperator{T}(m::Int64, n::Int64) where {T} =
    MatrixOperator{T}(
        m,
        n,
        nothing,
        new_id()
    )

Domain(A::MatrixOperator) = A.n
Range(A::MatrixOperator) = A.m

id(A::MatrixOperator) = A.id

function init(A::MatrixOperator{T}, pv::Optional{ParameterVector} = nothing) where {T}
    θ = rand(T, A.m, A.n)
    if !isnothing(pv)
        pv[A.id] = θ
    end
    return [θ]
end

param(A::MatrixOperator) = A.θ
nparam(A::MatrixOperator) = 1

(A::MatrixOperator{T})(pv::ParameterVector{T}) where {T} =
    MatrixOperator{T}(
        A.m,
        A.n,
        pv[A.id],
        A.id
    )

*(A::MatrixOperator{T}, x::V) where {T,V<:AbstractVector{T}} = A.θ*x
*(A::LinearOperatorAdjoint{T,T,MatrixOperator{T}}, x::V) where {T,V<:AbstractVector{T}} = A.inner.θ'*x