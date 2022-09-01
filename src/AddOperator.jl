struct AddOperator{D,R,L} <: Operator{D,R,L,NonParametric}
    lhs::Operator{D,R,<:Linearity,<:Parametricity}
    rhs::Operator{D,R,<:Linearity,<:Parametricity}
end

function +(lhs::Operator{D,R,<:Linearity,<:Parametricity}, rhs::Operator{D,R,<:Linearity,<:Parametricity}) where {D,R}
    @assert Domain(lhs) == Domain(rhs)
    @assert Range(lhs) == Range(rhs)
    return AddOperator{D,R,NonLinear}(lhs, rhs)
end

function +(lhs::Operator{D,R,Linear,<:Parametricity}, rhs::Operator{D,R,Linear,<:Parametricity}) where {D,R}
    @assert Domain(lhs) == Domain(rhs)
    @assert Range(lhs) == Range(rhs)
    AddOperator{D,R,Linear}(lhs, rhs)
end

Domain(A::AddOperator)  = Domain(A.lhs)
Range(A::AddOperator)   = Range(A.rhs)
nparams(A::AddOperator) = nparams(A.lhs) + nparams(A.rhs)
init(A::AddOperator)    = [init(A.lhs)..., init(A.rhs)...]
id(A::AddOperator)      = "[$(id(A.lhs))]_add_[$(id(A.rhs))]"

adjoint(A::AddOperator{D,R,Linear}) where {D,R} = AddOperator{R,D,Linear}(adjoint(A.lhs), adjoint(A.rhs))

(A::AddOperator{D,R,L})(x::V) where {D,R,L,V<:AbstractVector{D}} = A.lhs(x) + A.rhs(x)
(A::AddOperator{D,R,L})(θ::Vector{<:AbstractArray}) where {D,R,L} =
    AddOperator{D,R,L}(A.lhs(θ[1:nparams(A.lhs)]), A.rhs(θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)]))