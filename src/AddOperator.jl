import Base.+
import Base.*

struct AddOperator{D,R,LHS<:AbstractOperator{D,R},RHS<:AbstractOperator{D,R}} <: AbstractOperator{D,R}
    lhs::LHS
    rhs::RHS
    id::Any
end

function +(lhs::LHS, rhs::RHS) where
    {
        D,
        R,
        LHS<:AbstractOperator{D,R},
        RHS<:AbstractOperator{D,R}
    }
    @assert Range(lhs) == Range(rhs)
    return AddOperator{D,R,LHS,RHS}(
        lhs,
        rhs,
        "$(id(lhs))_add_$(id(rhs))"
    )
end

struct AddLinearOperator{D,R,LHS<:AbstractLinearOperator{D,R},RHS<:AbstractLinearOperator{D,R}} <: AbstractLinearOperator{D,R}
    lhs::LHS
    rhs::RHS
    id::Any
end

function +(lhs::LHS, rhs::RHS) where
    {
        D,
        R,
        LHS<:AbstractLinearOperator{D,R},
        RHS<:AbstractLinearOperator{D,R}
    }
    @assert Range(lhs) == Range(rhs)
    return AddLinearOperator{D,R,LHS,RHS}(
        lhs,
        rhs,
        "$(id(lhs))_add_$(id(rhs))"
    )
end

for op in [:AddOperator, :AddLinearOperator]
    @eval begin
        Domain(A::$op) = Domain(A.lhs)
        Range(A::$op) = Range(A.lhs)
        params(A::$op) = [params(A.lhs)..., params(A.rhs)...]
        nparams(A::$op) = nparams(A.lhs) + nparams(A.rhs)
        init(A::$op, pc::Optional{ParameterContainer} = nothing) = [init(A.lhs, pc)..., init(A.rhs, pc)...]
        id(A::$op) = A.id
    end
end

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function (A::AddOperator{D,R,LHS,RHS})(x::V, θs::Vararg{<:AbstractArray}) where
            {
                D,
                R,
                LHS<:AbstractOperator{D,R},
                RHS<:AbstractOperator{D,R},
                V<:$vectype{D}
            }
            return A.lhs(x, θs...) + A.rhs(x, θs...)
        end

        function (A::AddOperator{D,R,LHS,RHS})(x::V, pc::ParameterContainer) where
            {
                D,
                R,
                LHS<:AbstractOperator{D,R},
                RHS<:AbstractOperator{D,R},
                V<:$vectype{D}
            }
            return A.lhs(x, pc) + A.rhs(x, pc)
        end
    end
end

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function *(A::AddLinearOperator{D,R,LHS,RHS}, x::V) where
            {
                D,
                R,
                LHS<:AbstractLinearOperator{D,R},
                RHS<:AbstractLinearOperator{D,R},
                V<:$vectype{D}
            }
            return A.lhs*x + A.rhs*x
        end

        function *(A::LinearOperatorAdjoint{D,R,AddLinearOperator{D,R,LHS,RHS}}, x::V) where
            {
                D,
                R,
                LHS<:AbstractLinearOperator{D,R},
                RHS<:AbstractLinearOperator{D,R},
                V<:$vectype{R}
            }
            return A.lhs'*x + A.rhs'*x
        end
    end
end

function (A::AddLinearOperator{D,R,LHS,RHS})(θs::Vararg{<:AbstractArray}) where
    {
        D,
        R,
        LHS<:AbstractLinearOperator{D,R},
        RHS<:AbstractLinearOperator{D,R}
    }
    return A.lhs(θs...) + A.rhs(θs...)
end

function (A::AddLinearOperator{D,R,LHS,RHS})(pc::ParameterContainer) where
    {
        D,
        R,
        LHS<:AbstractLinearOperator{D,R},
        RHS<:AbstractLinearOperator{D,R}
    }
    return A.lhs(pc) + A.rhs(pc)
end