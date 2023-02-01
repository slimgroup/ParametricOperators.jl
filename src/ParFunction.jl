export ParFunction, ParLinearFunction, ParActivationFunction

"""
Opaque parametric function type holding a function F: Dⁿ ⟶ Rᵐ.
"""
struct ParFunction{D,R,L,P,F<:Function} <: ParOperator{D,R,L,P,External}
    f::F
    n::Int
    m::Int
    nparams::Int

    ParFunction(f::Function, n::Int, m::Int, D::Type, R::Type; L::Type{<:Linearity} = NonLinear, nparams::Int = 0) =
        new{D,R,L,(nparams > 0 ? Parametric : NonParametric),typeof(f)}(f, n, m, nparams)
end

Domain(A::ParFunction) = A.n
Range(A::ParFunction) = A.m
nparams(A::ParFunction) = A.nparams

ParLinearFunction(f, n::Int, m::Int, D::Type, R::Type; nparams::Int = 0) = ParFunction(f, n, m, D, R; L=Linear, nparams=nparams)
ParActivationFunction(f, n::Int, D::Type) = ParFunction(f, n, n, D, D; L=NonLinear, nparams=0)

(A::ParFunction{D,R,L,NonParametric,F})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.f(x)
(A::ParParameterized{D,R,L,ParFunction{D,R,L,Parametric,F}})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.f(x, A.params...)