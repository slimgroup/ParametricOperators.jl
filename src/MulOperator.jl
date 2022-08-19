import Base.*

struct MulOperator{D,R1,R2,LHS<:AbstractLinearOperator{R1,R2},RHS<:AbstractLinearOperator{D,R1}} <: AbstractLinearOperator{D,R2}
    lhs::LHS
    rhs::RHS
    id::Any
end

function *(lhs::LHS, rhs::RHS) where
    {
        D,
        R1,
        R2,
        LHS<:AbstractOperator{R1,R2},
        RHS<:AbstractOperator{D,R1}
    }
    @assert Domain(lhs) == Range(rhs)
    return MulOperator{D,R1,R2,LHS,RHS}(
        lhs,
        rhs,
        "$(id(lhs))_mul_$(id(rhs))"
    )
end

Domain(A::MulOperator) = Domain(A.lhs)
Range(A::MulOperator) = Range(A.lhs)
params(A::MulOperator) = [params(A.lhs)..., params(A.rhs)...]
nparams(A::MulOperator) = nparams(A.lhs) + nparams(A.rhs)
init(A::MulOperator, pc::Optional{ParameterContainer} = nothing) = [init(A.lhs, pc)..., init(A.rhs, pc)...]
id(A::MulOperator) = A.id

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function *(A::MulOperator{D,R1,R2,LHS,RHS}, x::V) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractOperator{R1,R2},
                RHS<:AbstractOperator{D,R1},
                V<:$vectype{D}
            }
            return A.lhs*(A.rhs*x)
        end
    end

    @eval begin
        function *(A::LinearOperatorAdjoint{D,R2,M}, x::V) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractOperator{R1,R2},
                RHS<:AbstractOperator{D,R1},
                M<:MulOperator{D,R1,R2,LHS,RHS},
                V<:$vectype{R2}
            }
            return A.inner.rhs'*(A.inner.lhs'*x)
        end
    end
end

