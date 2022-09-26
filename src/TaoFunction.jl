export TaoFunction

struct TaoFunction{D,R,L,P,F<:Function} <: TaoOperator{D,R,L,P,External}
    m::Int64
    n::Int64
    f::F
    id::ID
end

const TaoLinearFunction{D,R,P,F} = TaoFunction{D,R,Linear,P,F}

TaoFunction{D,R,L,P}(m::Int64, n::Int64, f::F) where {D,R,L,P,F<:Function} =
    TaoFunction{D,R,L,P,F}(m, n, f, uuid4(GLOBAL_RNG))

Domain(F::TaoFunction) = F.m
Range(F::TaoFunction) = F.n
id(F::TaoFunction) = F.id

(G::TaoFunction{D,R,L,P,F})(x::X) where {D,R,L,F,P<:NoAcceptParams,X<:AbstractVector{D}} = G.f(x)
(G::TaoParameterized{D,R,L,TaoFunction{D,R,L,Parametric,F},T,V})(x::X) where {D,R,L,F,X<:AbstractVector{D},T,V<:AbstractVector{T}} = G.op.f(x, G.θ)

# For now, assume that any function which is composed with a TaoOperator is
# elementwise and nonlinear (i.e. "activation functions")
function ∘(f::F1, F::F2) where {D,R,F1<:Function,F2<:TaoOperator{D,R,<:Linearity,<:Parametricity,<:NodeType}}
    G = TaoFunction{D,R,NonLinear,NonParametric}(Range(F), Range(F), f)
    return G ∘ F
end