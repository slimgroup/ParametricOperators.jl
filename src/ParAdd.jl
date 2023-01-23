export ParAdd

"""
Addition of operators.
"""
struct ParAdd{D,R,L,P,F,N} <: ParOperator{D,R,L,P,Internal}
    ops::F
    function ParAdd(ops...)
        ops = collect(ops)
        N = length(ops)

        if N == 1
            return ops[1]
        end

        @ignore_derivatives begin
            @assert allequal(map(Domain, ops))
            @assert allequal(map(Range, ops))
            @assert allequal(map(DDT, ops))
            @assert allequal(map(RDT, ops))
        end

        L = foldl(promote_linearity, map(linearity, ops))
        P = foldl(promote_parametricity, map(parametricity, ops))
        return new{DDT(ops[1]),RDT(ops[1]),L,P,typeof(ops),N}(ops)
    end
end

+(ops::ParOperator...) = ParAdd(ops...)
+(A::ParAdd, op::ParOperator) = ParAdd(A.ops..., op)
+(op::ParOperator, A::ParAdd) = ParAdd(op, A.ops...)
+(A::ParAdd, B::ParAdd) = ParAdd(A.ops..., B.ops...)

Domain(A::ParAdd) = Domain(A.ops[1])
Range(A::ParAdd) = Range(A.ops[1])
children(A::ParAdd) = A.ops
from_children(::ParAdd, cs) = ParAdd(cs...)
adjoint(A::ParAdd) = ParAdd(map(adjoint, A.ops)...)

(A::ParAdd{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractVector{D}} = mapreduce(op -> op(x), +, A.ops)
(A::ParAdd{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractMatrix{D}} = mapreduce(op -> op(x), +, A.ops)
*(x::X, A::ParAdd{D,R,L,<:Applicable,F,N}) where {D,R,L,F,N,X<:AbstractMatrix{R}} = mapreduce(op -> x*op, +, A.ops)
