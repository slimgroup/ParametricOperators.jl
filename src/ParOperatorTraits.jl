export linearity, parametricity
export DDT, RDT, Domain, Range, nparams, children, id, init

linearity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = L
parametricity(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = P

DDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = D
RDT(::ParOperator{D,R,L,P,T}) where {D,R,L,P,T} = R

Domain(A::ParAdjoint) = Range(A.op)
Domain(A::ParParameterized) = Domain(A.op)

Range(A::ParAdjoint) = Domain(A.op)
Range(A::ParParameterized) = Range(A.op)

nparams(::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = 0
nparams(A::ParOperator{D,R,L,P,Internal}) where {D,R,L,P<:Union{Parametric,Parameterized}} = sum(map(nparams, children(A)))

children(::ParOperator{D,R,L,P,External}) where {D,R,L,P} = []
children(A::ParAdjoint) = [A.op]
children(A::ParParameterized) = [A.op]

id(A::ParAdjoint) = "adjoint_[$(id(A.op))]"
id(A::ParParameterized) = "parameterized_[$(id(A.op))]"

init(::ParOperator{D,R,L,NonParametric,T}) where {D,R,L,T} = []
init(A::ParOperator{D,R,L,Parametric,Internal}) where {D,R,L} = vcat(map(init, filter(op -> parametricity(op) == Parametric, children(A)))...)

adjoint(A::ParOperator{D,R,Linear,P,T}) where {D,R,P,T} = ParAdjoint(A)
adjoint(A::ParAdjoint) = A.op

(A::ParOperator{D,R,L,Parametric,External})(θ::AbstractVector{<:Number}) where {D,R,L} = ParParameterized(A, θ)
(A::ParAdjoint{D,R,Parametric,F})(θ::AbstractVector{<:Number}) where {D,R,F} = ParAdjoint(A.op(θ))

*(A::ParOperator{D,R,Linear,<:Applicable,T}, x::X) where {D,R,T,X<:AbstractVecOrMat{D}} = A(x)
(A::ParOperator{D,R,L,<:Applicable,T})(x::X) where {D,R,L,T,X<:AbstractMatrix{D}} =
    mapreduce(A, hcat, eachcol(x))

function ChainRulesCore.rrule(A::F, x::X) where {D,R,T,F<:ParOperator{D,R,Linear,NonParametric,T},X<:AbstractVecOrMat{D}}
    y = A(x)
    function pullback(Δy)
        return NoTangent(), @thunk(A'(Δy))
    end
    return y, pullback
end