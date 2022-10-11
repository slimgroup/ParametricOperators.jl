export ParKron, ⊗

struct ParKron{D,R,P,F1,F2} <: ParLinearOperator{D,R,P,Internal}
    lhs::F1
    rhs::F2
    m::Int64
    n::Int64
    shape_in::NTuple{2, Int64}
    shape_out::NTuple{2, Int64}
    id::ID

    function ParKron(
        lhs::ParLinearOperator{D,R,<:Parametricity,<:ASTLocation},
        rhs::ParLinearOperator{D,R,<:Parametricity,<:ASTLocation}    
    ) where {D,R}
        P = promote_parametricity(parametricity(lhs), parametricity(rhs))
        shape_in = (Domain(rhs), Domain(lhs))
        shape_out = (Range(rhs), Range(lhs))
        m = prod(shape_out)
        n = prod(shape_in)
        return new{D,R,P,typeof(lhs),typeof(rhs)}(lhs, rhs, m, n, shape_in, shape_out, uuid4(GLOBAL_RNG))
    end
end

⊗(lhs::ParLinearOperator, rhs::ParLinearOperator) = ParKron(lhs, rhs)

Domain(A::ParKron) = A.n
Range(A::ParKron) = A.m
children(A::ParKron) = [A.lhs, A.rhs]
id(A::ParKron) = A.id

adjoint(A::ParKron{D,R,P,F1,F2}) where {D,R,P,F1,F2} = ParKron(adjoint(A.lhs), adjoint(A.rhs))

(A::ParKron{D,R,P,F1,F2})(θ) where
{
    D,
    R,
    P<:Parametric,
    F1<:ParLinearOperator{D,R,Parametric,<:ASTLocation},
    F2<:ParLinearOperator{D,R,Parametric,<:ASTLocation}
} = ParKron(A.lhs(θ[1:nparams(A.lhs)]), A.rhs(θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)]))

(A::ParKron{D,R,P,F1,F2})(θ) where
{
    D,
    R,
    P<:Parametric,
    F1<:ParLinearOperator{D,R,Parametric,<:ASTLocation},
    F2<:ParLinearOperator{D,R,<:Applicable,<:ASTLocation}
} = ParKron(A.lhs(θ), A.rhs)

(A::ParKron{D,R,P,F1,F2})(θ) where
{
    D,
    R,
    P<:Parametric,
    F1<:ParLinearOperator{D,R,<:Applicable,<:ASTLocation},
    F2<:ParLinearOperator{D,R,Parametric,<:ASTLocation}
} = ParKron(A.lhs, A.rhs(θ))

function (A::ParKron{D,R,P,F1,F2})(x::X) where {D,R,P<:Applicable,F1,F2,X<:AbstractVector{D}}
    xr = reshape(x, A.shape_in)
    yr = A.rhs*xr*A.lhs
    return vec(yr)
end

function kron_children(A::ParKron)
    cs_lhs = Base.typename(typeof(A)) == Base.typename(typeof(A.lhs)) ? kron_children(A.lhs) : [A.lhs]
    cs_rhs = Base.typename(typeof(A)) == Base.typename(typeof(A.rhs)) ? kron_children(A.rhs) : [A.rhs]
    return [cs_lhs..., cs_rhs...]
end