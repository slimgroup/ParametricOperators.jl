struct BiasOperator{T} <: AbstractLinearOperator{T,T}
    n::Int64
    θ::Optional{Vector{T}}
    id::Any
end

BiasOperator{T}(n::Int64) where {T} = BiasOperator{T}(
    n,
    nothing,
    new_id()
)

Domain(A::BiasOperator) = A.n
Range(A::BiasOperator) = A.n

id(A::BiasOperator) = A.id

function init(A::BiasOperator{T}, pv::Optional{ParameterVector} = nothing) where {T}
    θ = rand(T, A.n)
    if !isnothing(pv)
        pv[A.id] = θ
    end
    return [θ]
end

param(A::BiasOperator) = A.θ
nparam(A::BiasOperator) = 1

(A::BiasOperator{T})(pv::ParameterVector{T}) where {T} =
    BiasOperator{T}(
        A.n,
        pv[A.id],
        A.id
    )

*(A::BiasOperator{T}, ::V) where {T,V<:AbstractVector{T}} = A.θ
*(A::LinearOperatorAdjoint{T,T,BiasOperator{T}}, ::V) where {T,V<:AbstractVector{T}} = A.inner.θ