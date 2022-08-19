import Base.broadcasted

struct ElementwiseOperator{D,R,F<:Function,O<:AbstractOperator{D,R}} <: AbstractOperator{D,R}
    f::F
    op::O
    id::Any
end

function Base.broadcasted(f::F, op::O) where
    {
        D,
        R,
        F<:Function,
        O<:AbstractOperator{D,R}
    }
    return ElementwiseOperator{D,R,F,O}(
        f,
        op,
        "$(nameof(f))_$(id(op))"
    )
end

Domain(E::ElementwiseOperator) = Domain(E.op)
Range(E::ElementwiseOperator) = Range(E.op)
params(E::ElementwiseOperator) = params(E.op)
nparams(E::ElementwiseOperator) = nparams(E.op)
id(E::ElementwiseOperator) = E.id
init(E::ElementwiseOperator, pc::Optional{ParameterContainer} = nothing) = init(E.op, pc)

for vectype in [:AbstractVector, :AbstractVecOrMat]
    @eval begin
        function (E::ElementwiseOperator{D,R,F,O})(x::V, θs::Vararg{<:AbstractArray}) where
            {
                D,
                R,
                F<:Function,
                O<:AbstractOperator{D,R},
                V<:$vectype{D}
            }
            return (E.f).(E.op(x, θs...))
        end

        function (E::ElementwiseOperator{D,R,F,O})(x::V, pc::ParameterContainer) where
            {
                D,
                R,
                F<:Function,
                O<:AbstractOperator{D,R},
                V<:$vectype{D}
            }
            return (E.f).(E.op(x, pc))
        end

        function (E::ElementwiseOperator{D,R,F,O})(x::V, θs::Vararg{<:AbstractArray}) where
            {
                D,
                R,
                F<:Function,
                O<:AbstractLinearOperator{D,R},
                V<:$vectype{D}
            }
            return (E.f).(E.op(θs...)*x)
        end

        function (E::ElementwiseOperator{D,R,F,O})(x::V, pc::ParameterContainer) where
            {
                D,
                R,
                F<:Function,
                O<:AbstractLinearOperator{D,R},
                V<:$vectype{D}
            }
            return (E.f).(E.op(pc)*x)
        end
    end
end
