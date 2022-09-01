struct CompositionOperator{D,T,R} <: Operator{D,R,NonLinear,NonParametric}
    lhs::Operator{T,R,<:Linearity,<:Parametricity}
    rhs::Operator{D,T,<:Linearity,<:Parametricity}
end

function ∘(lhs::Operator{T,R,<:Linearity,<:Parametricity}, rhs::Operator{D,T,<:Linearity,<:Parametricity}) where {D,T,R}
    @assert Domain(lhs) == Range(rhs)
    return CompositionOperator{D,R,T}(lhs, rhs)
end

function ∘(lhs::Operator{T,R,Linear,<:Parametricity}, rhs::Operator{D,T,Linear,<:Parametricity}) where {D,T,R}
    return MulOperator{D,T,R}(lhs, rhs)
end

function ∘(f::Function, rhs::Operator{D,R,<:Linearity,<:Parametricity}) where {D,R}
    return FunctionOperator{R,R,NonLinear}(Range(rhs), Range(rhs), f) ∘ rhs
end

Domain(C::CompositionOperator)  = Domain(C.rhs)
Range(C::CompositionOperator)   = Range(C.lhs)
nparams(C::CompositionOperator) = nparams(C.lhs) + nparams(C.rhs)
init(C::CompositionOperator)    = [init(C.lhs)..., init(C.rhs)...]
id(C::CompositionOperator)      = "[$(id(C.lhs))]_comp_[$(id(C.rhs))]"

(C::CompositionOperator{D,T,R})(x::V) where {D,T,R,V<:AbstractVector{D}} = C.lhs(C.rhs(x))
(C::CompositionOperator{D,T,R})(θ::Vector{<:AbstractArray}) where {D,T,R} =
    CompositionOperator{D,T,R}(C.lhs(θ[1:nparams(C.lhs)]), C.rhs(θ[nparams(C.lhs)+1:nparams(C.lhs)+nparams(C.rhs)]))