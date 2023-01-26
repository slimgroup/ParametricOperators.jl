export ParCompose

"""
Describes whether parameters should be passed left to right or right to left.
"""
abstract type ParamOrder end
struct LeftFirst <: ParamOrder end
struct RightFirst <: ParamOrder end

struct ParCompose{D,R,L,P,F,N,O<:ParamOrder} <: ParOperator{D,R,L,P,Internal}
    ops::F
    function ParCompose(ops...; order=LeftFirst)
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

        return new{D,R,L,P,typeof(ops),N,order}(ops)
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
∘(A::ParCompose, op::ParOperator) = ParCompose(A.ops..., op)
∘(op::ParOperator, A::ParCompose) = ParCompose(op, A.ops...)
∘(A::ParCompose, B::ParCompose) = ParCompose(A.ops..., B.ops...)
*(ops::ParLinearOperator...) = ∘(ops...)

Domain(A::ParCompose{D,R,L,P,F,N,O}) where {D,R,L,P,F,N,O} = Domain(A.ops[N])
Range(A::ParCompose{D,R,L,P,F,N,O}) where {D,R,L,P,F,N,O} = Range(A.ops[1])
children(A::ParCompose) = A.ops
from_children(::ParCompose, cs) = ParCompose(cs...)

adjoint(A::ParCompose{D,R,Linear,P,F,N,LeftFirst}) where {D,R,P,F,N} = ParCompose(reverse(map(adjoint, A.ops))...; order=RightFirst)
adjoint(A::ParCompose{D,R,Linear,P,F,N,RightFirst}) where {D,R,P,F,N} = ParCompose(reverse(map(adjoint, A.ops))...; order=LeftFirst)

function (A::ParCompose{D,R,L,Parametric,F,N,LeftFirst})(params) where {D,R,L,F,N}
    param_ranges = cumranges([nparams(c) for c in children(A)])
    cs_out = [parametricity(c) == Parametric ? c(params[r]) : c for (c, r) in zip(children(A), param_ranges)]
    return from_children(A, cs_out)
end

function (A::ParCompose{D,R,L,Parametric,F,N,RightFirst})(params) where {D,R,L,F,N}
    param_ranges = cumranges([nparams(c) for c in children(A)])
    cs_out = [parametricity(c) == Parametric ? c(params[r]) : c for (c, r) in zip(children(A), reverse(param_ranges))]
    return from_children(A, cs_out)
end

function (A::ParCompose{D,R,L,<:Applicable,F,N,O})(x::X) where {D,R,L,F,N,O,X<:AbstractVector{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function (A::ParCompose{D,R,L,<:Applicable,F,N,O})(x::X) where {D,R,L,F,N,O,X<:AbstractMatrix{D}}
    for i in 1:N
        x = A.ops[N-i+1](x)
    end
    return x
end

function *(x::X, A::ParCompose{D,R,Linear,<:Applicable,F,N,O}) where {D,R,F,N,O,X<:AbstractMatrix{R}}
    for i in 1:N
        x = x*A.ops[i]
    end
    return x
end
