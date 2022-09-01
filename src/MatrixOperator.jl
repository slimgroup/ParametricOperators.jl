struct MatrixOperator{T} <: Operator{T,T,Linear,NonParametric}
    m::Int64
    n::Int64
    id::Any
end

MatrixOperator{T}(m::Int64, n::Int64) where {T} =
    MatrixOperator{T}(
        m,
        n,
        uid()
    )

Domain(A::MatrixOperator) = A.n
Range(A::MatrixOperator)  = A.m
nparams(::MatrixOperator) = 1
init(A::MatrixOperator{T}) where {T} = [T(1)/T(A.m*A.n)*randn(T, A.m, A.n)]
id(A::MatrixOperator) = A.id

(A::Parameterized{T,T,Linear,MatrixOperator{T}})(x::V) where {T<:Number,V<:AbstractVector{T}} = A.θ[1]*x
(A::Parameterized{T,T,Linear,Adjoint{T,T,NonParametric,MatrixOperator{T}}})(x::V) where {T<:Number,V<:AbstractVector{T}} = A.θ[1]'*x