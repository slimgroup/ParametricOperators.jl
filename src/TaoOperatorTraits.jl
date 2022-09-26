export DDT, RDT, Domain, Range, children, nparams, init

linearity(::TaoOperator{D,R,L,P}) where {D,R,L,P} = L
parametricity(::TaoOperator{D,R,L,P}) where {D,R,L,P} = P

DDT(::TaoOperator{D,R,L,P}) where {D,R,L,P} = D
RDT(::TaoOperator{D,R,L,P}) where {D,R,L,P} = R

Domain(A::TaoAdjoint) = Range(A.op)
Domain(P::TaoParameterized) = Domain(P.op)

Range(A::TaoAdjoint) = Domain(A.op)
Range(P::TaoParameterized) = Range(P.op)

children(A::TaoAdjoint) = [A.op]
children(P::TaoParameterized) = [P.op]

nparams(F::TaoOperator{D,R,L,<:NoAcceptParams,T}) where {D,R,L,T} = 0
nparams(F::TaoOperator{D,R,L,Parametric,Internal}) where {D,R,L} =
    sum(map(nparams, children(F)))

init(::TaoOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = []
init(A::TaoAdjoint) = [init(A.op)]
init(P::TaoParameterized) = [init(P.op)]

adjoint(A::TaoOperator{D,R,Linear,P,External}) where {D,R,P} = TaoAdjoint(A)
adjoint(P::TaoParameterized) = TaoAdjoint(P)
adjoint(A::TaoAdjoint) = A.op

(F::TaoOperator{D,R,L,Parametric,External})(θ::V) where {D,R,L,V<:AbstractVector} = TaoParameterized(F, θ)
(F::TaoOperator{D,R,L,Parametric,<:NodeType})(x::X, θ::V) where {D,R,L,T<:Number,X<:AbstractVector{D},V<:AbstractVector{T}} = F(θ)(x)
*(F::TaoOperator{D,R,Linear,P,T}, x::AbstractVecOrMat{D}) where {D,R,P<:NoAcceptParams,T} = F(x)
(A::TaoAdjoint{D,R,Parametric,F})(θ::V) where {D,R,F,V<:AbstractVector} = adjoint(A.op(θ))

function ChainRulesCore.rrule(A::TaoOperator{D,R,Linear,NonParametric,T}, x::AbstractVector{D}) where {D,R,T}
    y = A(x)
    function pullback(dy)
        dx = @thunk(A'(dy))
        return NoTangent(), dx
    end
    return y, pullback
end