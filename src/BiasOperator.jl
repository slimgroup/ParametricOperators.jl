import Base.*

"""
Bias operator.
"""
struct BiasOperator{T} <: AbstractLinearOperator{T,T}
    θ::Optional{Vector{T}}
    n::Int64
    id::Any
end

"""
Bias constructor.
"""
BiasOperator{T}(n::Int64) where {T} =
    BiasOperator{T}(
        nothing,
        n,
        uid()
    )

# Bias operator impls
Domain(A::BiasOperator) = A.n
Range(A::BiasOperator) = A.n
params(A::BiasOperator) = [A.θ]
nparams(A::BiasOperator) = 1

function init(A::BiasOperator{T}, pc::Optional{ParameterContainer} = nothing) where {T}
    θ = zeros(T, A.n)
    if !isnothing(pc)
        pc[A.id] = θ
    end
    return [θ]
end

id(A::BiasOperator) = A.id

*(A::BiasOperator{T}, ::V) where {T,V<:AbstractVector{T}} = A.θ
*(A::LinearOperatorAdjoint{T,T,BiasOperator{T}}, ::Any) where {T} = A.inner.θ

(A::BiasOperator{T})(V::Vector{T}) where {T} =
    BiasOperator{T}(
        V,
        A.n,
        A.id
    )

(A::BiasOperator{T})(pc::ParameterContainer) where {T} =
    BiasOperator{T}(
        pc[A.id][1],
        A.n,
        A.id
    )