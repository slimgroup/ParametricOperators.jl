import Base.∘
import Base.broadcasted

abstract type AbstractOperator{D,R} end

Domain(F::AbstractOperator) = throw(TaoException("Domain() is not implemented for $(typeof(F))"))
Range(F::AbstractOperator) = throw(TaoException("Domain() is not implemented for $(typeof(F))"))
ddt(::AbstractOperator{D,R}) where {D,R} = D
rdt(::AbstractOperator{D,R}) where {D,R} = R

id(F::AbstractOperator) = throw(TaoException("id() is not implemented for $(typeof(F))"))
init(F::AbstractOperator, ::Optional{ParameterVector} = nothing) = throw(TaoException("init() is not implemented for $(typeof(F))"))
param(F::AbstractOperator) = throw(TaoException("param() is not implemented for $(typeof(F))"))
nparam(F::AbstractOperator) = throw(TaoException("nparam() is not implemented for $(typeof(F))"))
(F::AbstractOperator)(::ParameterVector) = throw(TaoException("vector parameterization is not implemented for $(typeof(F))"))
(F::AbstractOperator)(::AbstractVector, ::ParameterVector) = throw(TaoException("call() is not implemented for $(typeof(F))"))

DomainShape(F::AbstractOperator) = Domain(F)
RangeShape(F::AbstractOperator) = Range(F)

struct ElementwiseOperator{D,R,F<:AbstractOperator{D,R}} <: AbstractOperator{D,R}
    f::Any
    op::F
    id::Any
end

function Base.broadcasted(f, op::F) where {D,R,F<:AbstractOperator{D,R}}
    return ElementwiseOperator{D,R,F}(
        f,
        op,
        "$(typeof(f))_elementwise_$(id(op))"
    )
end

Domain(F::ElementwiseOperator) = Domain(F.op)
Range(F::ElementwiseOperator) = Range(F.op)

id(F::ElementwiseOperator) = F.id
init(F::ElementwiseOperator, pv::Optional{ParameterVector} = nothing) = init(F.op, pv)
param(F::ElementwiseOperator) = param(F.op)
nparam(F::ElementwiseOperator) = nparam(F.op)

(E::ElementwiseOperator{D,R,F})(pv::ParameterVector) where {D,R,F<:AbstractOperator{D,R}} =
    ElementwiseOperator{D,R,F}(
        E.f,
        E.op(pv),
        E.id
    )

function (E::ElementwiseOperator{D,R,F})(x::AbstractVector{D}, pv::ParameterVector) where {D,R,F<:AbstractOperator{D,R}}
    return (E.f).(E.op(x, pv))
end

struct CompositionOperator{D,R1,R2,F<:AbstractOperator{D,R1},G<:AbstractOperator{R1,R2}} <: AbstractOperator{D,R2}
    g::G
    f::F
    id::Any
end

function ∘(g::G, f::F) where {D,R1,R2,F<:AbstractOperator{D,R1},G<:AbstractOperator{R1,R2}}
    return CompositionOperator{D,R1,R2,F,G}(
        g,
        f,
        "$(id(g))_comp_$(id(f))"
    )
end

Domain(F::CompositionOperator) = Domain(F.f)
Range(F::CompositionOperator) = Range(F.g)

id(F::CompositionOperator) = F.id
init(F::CompositionOperator, pv::Optional{ParameterVector} = nothing) = [init(F.g, pv)..., init(F.f, pv)...]
param(F::CompositionOperator) = [param(F.g)..., param(F.f)...]
nparam(F::CompositionOperator) = nparam(F.g) + nparam(F.f)

(C::CompositionOperator{D,R1,R2,F,G})(pv::ParameterVector) where {D,R1,R2,F<:AbstractOperator{D,R1},G<:AbstractOperator{R1,R2}} =
    CompositionOperator{D,R1,R2,F,G}(
        C.g(pv),
        C.f(pv),
        C.id
    )

function (C::CompositionOperator{D,R1,R2,F,G})(x::AbstractVector{D}, pv::ParameterVector) where {D,R1,R2,F<:AbstractOperator{D,R1},G<:AbstractOperator{R1,R2}}
    return C.g(C.f(x, pv), pv)
end