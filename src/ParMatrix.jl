export ParMatrix

struct ParMatrix{T} <: ParLinearOperator{T,T,Parametric,FirstOrder}
    m::Int64
    n::Int64
    id::ID
    ParMatrix(m, n) = new{Float64}(m, n, uuid4(GLOBAL_RNG))
    ParMatrix(T, m, n) = new{T}(m, n, uuid4(GLOBAL_RNG))
end

Domain(A::ParMatrix) = A.n
Range(A::ParMatrix) = A.m
nparams(A::ParMatrix) = A.m*A.n
init(A::ParMatrix{T}) where {T} = T(1/nparams(A)).*rand(T, nparams(A))