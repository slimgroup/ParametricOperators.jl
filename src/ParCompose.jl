export ParCompose

"""
Composition operator.
"""
struct ParCompose{D,R,L,P,F,N} <: ParOperator{D,R,L,P,Internal}
    ops::F
    function ParCompose(ops...)
        ops = collect(ops)
        ops = collect(ops)
        N = length(ops)
        if N == 1
            return ops[1]
        end
        @ignore_derivatives begin
            for i in 1:N-1
                @assert Domain(ops[i]) == Range(ops[i+1])
                @assert DDT(ops[i]) == RDT(ops[i+1])
            end
        end

        D = DDT(ops[N])
        R = RDT(ops[1])
        L = foldl(promote_linearity, map(linearity, ops))
        P = foldl(promote_parametricity, map(parametricity, ops))

        return new{D,R,L,P,typeof(ops),length(ops)}(ops)
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
∘(A::ParCompose, op::ParOperator) = ParCompose(A.ops..., op)
∘(op::ParOperator, A::ParCompose) = ParCompose(op, A.ops...)
∘(A::ParCompose, B::ParCompose) = ParCompose(A.ops..., B.ops...)
*(ops::ParLinearOperator...) = ∘(ops...)


Domain(A::ParCompose{D,R,L,P,F,N}) where {D,R,L,P,F,N} = Domain(A.ops[N])
Range(A::ParCompose{D,R,L,P,F,N}) where {D,R,L,P,F,N} = Range(A.ops[1])
children(A::ParCompose) = A.ops
rebuild(::ParCompose, cs) = ParCompose(cs...)

adjoint(A::ParCompose{D,R,Linear,P,F,N}) where {D,R,P,F,N} = ParCompose(reverse(map(adjoint, A.ops))...)

function (A::ParCompose{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractVector{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function (A::ParCompose{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractMatrix{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function *(x::X, A::ParCompose{D,R,Linear,<:Applicable,F,N}) where {D,R,F,N,X<:AbstractMatrix{R}}
    for i in 1:N
        x = x*A.ops[i]
    end
    return x
end