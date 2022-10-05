export ParFunction

struct ParFunction{D,R,L,P,F<:Function} <: ParOperator{D,R,L,P,External}
    m::Int64
    n::Int64
    f::F
    id::ID
    ParFunction(D,R,L,P,m,n,f) = new{D,R,L,P,typeof(f)}(m, n, f, uuid4(GLOBAL_RNG))
end

Domain(F::ParFunction) = F.n
Range(F::ParFunction) = F.m
id(F::ParFunction) = F.id

(A::ParFunction{D,R,L,NonParametric,F})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.f(x)
(A::ParParameterized{D,R,L,ParFunction{D,R,L,Parametric,F},V})(x::X) where
    {D,R,L,F,V<:AbstractVector{D},X<:AbstractVector{D}} = A.op.f(x, A.θ)

∘(f::F, A::ParOperator{D,R,L,P,T}) where {F<:Function,D,R,L,P,T} =
    ParCompose(ParFunction(R,R,NonLinear,NonParametric,Range(A),Range(A),f), A)

∘(A::ParOperator{D,R,L,P,T}, f::F) where {F<:Function,D,R,L,P,T} =
    ParCompose(ParFunction(D,D,NonLinear,NonParametric,Domain(A),Domain(A),f), A)