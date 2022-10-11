export ParFunction, ParLinearFunction, ParLinearFunctional

struct ParFunction{D,R,L,P,F} <: ParOperator{D,R,L,P,External}
    f::F
    m::Int64
    n::Int64
    id::ID
    ParFunction(f::Function, n::Integer; D=Float64, R=Float64, L=NonLinear, P=NonParametric) =
        new{D,R,L,P,typeof(f)}(f, n, n, uuid4(GLOBAL_RNG))
end

ParLinearFunction(f::Function, n::Integer; D=Float64, R=Float64, P=NonParametric) =
    ParFunction(f, n; D=D, R=R, L=Linear, P=P)

Domain(A::ParFunction) = A.n
Range(A::ParFunction) = A.m
id(A::ParFunction) = A.id

(A::ParFunction{D,R,L,P,F})(x::X) where {D,R,L,P<:Applicable,F,X<:AbstractVector{D}} = A.f(x)
(A::ParParameterized{D,R,L,ParFunction{D,R,L,Parametric,F},V})(x::X) where
    {D,R,L,F,X<:AbstractVector{D},T,V<:AbstractVector{T}} = A.op.f(x, A.θ)