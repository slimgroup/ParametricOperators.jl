import Base.∘

struct CompositionOperator{D,R1,R2,LHS<:AbstractOperator{R1,R2},RHS<:AbstractOperator{D,R1}} <: AbstractOperator{D,R2}
    lhs::LHS
    rhs::RHS
    id::Any
end

function ∘(lhs::LHS, rhs::RHS) where {D,R1,R2,LHS<:AbstractOperator{R1,R2},RHS<:AbstractOperator{D,R1}}
    @assert Range(rhs) == Domain(lhs)
    return CompositionOperator{D,R1,R2,LHS,RHS}(
        lhs,
        rhs,
        "$(id(lhs))_comp_$(id(rhs))"
    )
end

function ∘(lhs::F, rhs::RHS) where {D,R,F<:Function,RHS<:AbstractOperator{D,R}}
    n = Range(rhs)
    m = Range(rhs)
    return CompositionOperator{D,R,R,FunctionOperator{D,R,F},RHS}(
        FunctionOperator{D,R}(lhs,n,m),
        rhs,
        "$(nameof(lhs))_comp_$(id(rhs))"
    )
end

Domain(C::CompositionOperator) = Domain(C.rhs)
Range(C::CompositionOperator) = Range(C.lhs)
params(C::CompositionOperator) = [params(C.lhs)..., params(C.rhs)...]
nparams(C::CompositionOperator) = nparams(C.lhs) + nparams(C.rhs)
id(C::CompositionOperator) = C.id
init(C::CompositionOperator, pc::Optional{ParameterContainer} = nothing) = [init(C.lhs, pc)..., init(C.rhs, pc)...]

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function (C::CompositionOperator{D,R1,R2,LHS,RHS})(x::V, θs::Vararg{<:AbstractArray}) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractOperator{R1,R2},
                RHS<:AbstractOperator{D,R1},
                V<:$vectype{D}
            }
            return C.lhs(C.rhs(x, θs...), θs...)
        end

        function (C::CompositionOperator{D,R1,R2,LHS,RHS})(x::V, pc::ParameterContainer) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractOperator{R1,R2},
                RHS<:AbstractOperator{D,R1},
                V<:$vectype{D}
            }
            return C.lhs(C.rhs(x, pc), pc)
        end

        function (C::CompositionOperator{D,R1,R2,LHS,RHS})(x::V, pc::ParameterContainer) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractLinearOperator{R1,R2},
                RHS<:AbstractOperator{D,R1},
                V<:$vectype{D}
            }
            return C.lhs(pc)*C.rhs(x, pc)
        end

        function (C::CompositionOperator{D,R1,R2,LHS,RHS})(x::V, pc::ParameterContainer) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractOperator{R1,R2},
                RHS<:AbstractLinearOperator{D,R1},
                V<:$vectype{D}
            }
            return C.lhs(C.rhs(pc)*x, pc)
        end

        function (C::CompositionOperator{D,R1,R2,LHS,RHS})(x::V, θs::Vararg{<:AbstractArray}) where
            {
                D,
                R1,
                R2,
                LHS<:AbstractLinearOperator{R1,R2},
                RHS<:AbstractLinearOperator{D,R1},
                V<:$vectype{D}
            }
            return C.lhs(pc)*(C.rhs(pc)*x)
        end
    end
end
