import Base.+

struct AddOperator{D,R,LHS<:AbstractLinearOperator{D,R},RHS<:AbstractLinearOperator{D,R}} <: AbstractLinearOperator{D,R}
    lhs::LHS
    rhs::RHS
end

function +(lhs::AbstractLinearOperator{D,R}, rhs::AbstractLinearOperator{D,R}) where {D,R}
    @assert Range(lhs) == Range(rhs)
    return AddOperator{D,R,typeof(lhs),typeof(rhs)}(
        lhs,
        rhs
    )
end

Domain(A::AddOperator) = Domain(A.lhs)
Range(A::AddOperator) = Range(A.rhs)
id(A::AddOperator) = "$(id(A.lhs))_add_$(id(A.rhs))"

function init(A::AddOperator, pv::Optional{ParameterVector} = nothing)
    θ_lhs = init(A.lhs, pv)
    θ_rhs = init(A.rhs, pv)
    return [θ_lhs..., θ_rhs...]
end

param(A::AddOperator) = [param(A.lhs)..., param(A.rhs)...]
nparam(A::AddOperator) = nparam(A.lhs) + nparam(A.rhs)

(A::AddOperator{D,R,LHS,RHS})(pv::ParameterVector) where {D,R,LHS<:AbstractLinearOperator{D,R},RHS<:AbstractLinearOperator{D,R}} =
    AddOperator{D,R,LHS,RHS}(
        A.lhs(pv),
        A.rhs(pv)
    )

*(A::AddOperator{D,R,RHS,LHS}, x::V) where
    {
        D,
        R,
        RHS<:AbstractLinearOperator{D,R},
        LHS<:AbstractLinearOperator{D,R},
        V<:AbstractVector{D}
    } = A.lhs*x + A.rhs*x

*(A::LinearOperatorAdjoint{D,R,AddOperator{D,R,RHS,LHS}}, x::V) where
    {
        D,
        R,
        RHS<:AbstractLinearOperator{D,R},
        LHS<:AbstractLinearOperator{D,R},
        V<:AbstractVector{D}
    } = A.lhs'*x + A.rhs'*x