
"""
Cache for adjoint of nonparametic linear operators. Reduces overhead of
recomputing adjoints for deeply nested expressions, Kronecker products,
etc.
"""
_OPERATOR_ADJOINT_CACHE = LRU{ParLinearOperator,ParLinearOperator}(maxsize = 256); # TODO: is there a correct value for maxsize?

function ChainRulesCore.rrule(A::F, x::X) where
    {D,R,T,F<:ParLinearOperator{D,R,NonParametric,T},X<:AbstractVector{D}}
    y = A(x)
    function pullback(dy)
        global _OPERATOR_ADJOINT_CACHE
        A_adj = get!(_OPERATOR_ADJOINT_CACHE, A, A')
        dx = A_adj*dy
        return NoTangent(), dx
    end
    return y, pullback
end

function ChainRulesCore.rrule(A::F, x::X) where
    {D,R,T,F<:ParLinearOperator{D,R,NonParametric,T},X<:AbstractMatrix{D}}
    y = A(x)
    function pullback(dy)
        global _OPERATOR_ADJOINT_CACHE
        A_adj = get!(_OPERATOR_ADJOINT_CACHE, A, A')
        dx = A_adj*dy
        return NoTangent(), dx
    end
    return y, pullback
end