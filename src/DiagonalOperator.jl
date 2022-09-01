struct DiagonalOperator{T} <: Operator{T,T,Linear,NonParametric}
    n::Int64
    id::Any
end

DiagonalOperator{T}(n::Int64) where {T} =
    DiagonalOperator{T}(
        n,
        uid()
    )

Domain(A::DiagonalOperator) = A.n
Range(A::DiagonalOperator)  = A.n
nparams(::DiagonalOperator) = 1
init(A::DiagonalOperator{T}) where {T} = [randn(T, A.n)]
id(A::DiagonalOperator) = A.id

(A::Parameterized{T,T,Linear,DiagonalOperator{T}})(x::V) where {T<:Number,V<:AbstractVector{T}} = A.θ[1].*x
(A::Parameterized{T,T,Linear,Adjoint{T,T,NonParametric,DiagonalOperator{T}}})(x::V) where {T<:Number,V<:AbstractVector{T}} = conj(A.θ[1]).*x