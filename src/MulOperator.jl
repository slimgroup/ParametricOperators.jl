struct MulOperator{D,T,R,O<:ParamOrdering} <: Operator{D,R,Linear,NonParametric}
    lhs::Operator{T,R,Linear,<:Parametricity}
    rhs::Operator{D,T,Linear,<:Parametricity}
end

function *(lhs::Operator{T,R,Linear,<:Parametricity}, rhs::Operator{D,T,Linear,<:Parametricity}) where {D,T,R}
    @assert Domain(lhs) == Range(rhs)
    return MulOperator{D,T,R,LeftFirst}(lhs, rhs)
end

Domain(A::MulOperator)  = Domain(A.rhs)
Range(A::MulOperator)   = Range(A.lhs)
nparams(A::MulOperator) = nparams(A.lhs) + nparams(A.rhs)
init(A::MulOperator)    = [init(A.lhs)..., init(A.rhs)...]
id(A::MulOperator)      = "[$(id(A.lhs))]_add_[$(id(A.rhs))]"

adjoint(A::MulOperator{D,T,R,LeftFirst}) where {D,T,R} = MulOperator{R,T,D,RightFirst}(adjoint(A.rhs), adjoint(A.lhs))
adjoint(A::MulOperator{D,T,R,RightFirst}) where {D,T,R} = MulOperator{R,T,D,LeftFirst}(adjoint(A.rhs), adjoint(A.lhs))

(A::MulOperator{D,T,R,<:ParamOrdering})(x::V) where {D,T,R,V<:AbstractVector{D}} = A.lhs*(A.rhs*x)
(A::MulOperator{D,T,R,LeftFirst})(θ::Vector{<:AbstractArray}) where {D,T,R} =
    MulOperator{D,T,R,LeftFirst}(A.lhs(θ[1:nparams(A.lhs)]), A.rhs(θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)]))
(A::MulOperator{D,T,R,RightFirst})(θ::Vector{<:AbstractArray}) where {D,T,R} =
    MulOperator{D,T,R,RightFirst}(A.lhs(θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)]), A.rhs(θ[1:nparams(A.lhs)]))