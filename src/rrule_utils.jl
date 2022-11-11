children_symbol(A) = :ops
param_symbol(A) = :θ

function extract_param_gradients!(A, ∂B, θs)
    for (op, ∂op) in zip(children(A), ∂B[children_symbol(A)])
        if order(op) == HigherOrder
            extract_param_gradients!(op, ∂op, θs)
        elseif order(op) == FirstOrder && parametricity(op) == Parametric
            push!(θs, ∂op[param_symbol(A)])
        end
    end
end

function ChainRulesCore.rrule(A::ParLinearOperator{D,R,NonParametric,O}, x::X) where {D,R,O,X<:AbstractVector{D}}
    y = A*x
    function pullback(∂y::Y) where {R,Y<:AbstractVector{R}}
        ∂x = @thunk(A'*∂y)
        return NoTangent(), ∂x
    end
    return y, pullback
end

function ChainRulesCore.rrule(A::ParLinearOperator{D,R,NonParametric,O}, x::X) where {D,R,O,X<:AbstractMatrix{D}}
    y = A*x
    function pullback(∂y::Y) where {R,Y<:AbstractMatrix{R}}
        ∂x = @thunk(A'*∂y)
        return NoTangent(), ∂x
    end
    return y, pullback
end