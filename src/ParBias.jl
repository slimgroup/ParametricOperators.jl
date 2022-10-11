export ParBias

struct ParBias{T} <: ParOperator{Nothing,T,NonLinear,Parametric,External}
    n::Int64
    id::ID
    ParBias(n; T = Float64) = new{T}(n, uuid4(GLOBAL_RNG))
end

Range(A::ParBias) = A.n
nparams(A::ParBias) = A.n
init(A::ParBias{T}) where {T} = zeros(T, nparams(A))
id(A::ParBias) = A.id

(A::ParBias{T})(::X, θ::V) where {T,X,V<:AbstractVector{T}} = θ