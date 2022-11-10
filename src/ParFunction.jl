export ParFunction

struct ParFunction{D,R,L,P,F<:Function} <: ParOperator{D,R,L,P,FirstOrder}
    f::F
    m::Int64
    n::Int64
    id::ID
    ParFunction(f; L=NonLinear, P=NonParametric) = new{Nothing,Nothing,L,P,typeof(f)}(f, nothing, nothing, uuid4(GLOBAL_RNG))
    ParFunction(f, m, n; L=NonLinear, P=NonParametric) = new{Nothing,Nothing,L,P,typeof(f)}(f, m, n, uuid4(GLOBAL_RNG))
    ParFunction(T, f, m, n; L=NonLinear, P=NonParametric) = new{T,T,L,P,typeof(f)}(f, m, n, uuid4(GLOBAL_RNG))
    ParFunction(D, R, f, m, n; L=NonLinear, P=NonParametric) = new{D,R,L,P,typeof(f)}(f, m, n, uuid4(GLOBAL_RNG))
end

ParLinearFunction(args...; kwargs...) = ParFunction(args...; L=Linear, kwargs...)

Domain(A::ParFunction) = A.n
Range(A::ParFunction) = A.m
id(A::ParFunction) = A.id

(A::ParFunction{D,R,L,NonParametric,F})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.f(x)
(A::ParFunction{D,R,L,NonParametric,F})(x::X) where {D,R,L,F,X<:AbstractMatrix{D}} = A.f(x)

# By default, assume functions are elementwise nonlinear
∘(A::ParOperator, f::Function) = A ∘ ParFunction(DDT(A), f, Domain(A), Domain(A)) 
∘(f::Function, A::ParOperator) = ParFunction(RDT(A), f, Range(A), Range(A)) ∘ A