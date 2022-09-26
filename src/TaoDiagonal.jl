export TaoDiagonal

struct TaoDiagonal{T} <: TaoOperator{T,T,Linear,Parametric,External}
    n::Int64
    id::ID
end

TaoDiagonal{T}(n::Int64) where {T} = TaoDiagonal{T}(n, uuid4(GLOBAL_RNG))

Domain(A::TaoDiagonal) = A.n
Range(A::TaoDiagonal) = A.n
nparams(A::TaoDiagonal) = A.n
id(A::TaoDiagonal) = A.id
init(A::TaoDiagonal{T}) where {T} = ones(T, nparams(A))

(A::TaoParameterized{T,T,Linear,TaoDiagonal{T},T,V})(x::X) where {T,V<:AbstractVector{T},X<:AbstractVecOrMat{T}} = A.θ.*x
(A::TaoAdjoint{T,T,NonParametric,TaoParameterized{T,T,Linear,TaoDiagonal{T},T,V}})(y::Y) where 
    {T,V<:AbstractVector{T},Y<:AbstractVecOrMat{T}} = conj(A.op.θ).*y