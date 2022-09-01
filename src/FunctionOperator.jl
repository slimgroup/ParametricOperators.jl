struct FunctionOperator{D,R,L} <: Operator{D,R,L,NonParametric}
    m::Int64
    n::Int64
    f::Function
end

Domain(F::FunctionOperator)  = F.n
Range(F::FunctionOperator)   = F.m
nparams(F::FunctionOperator) = 0
init(::FunctionOperator)     = Vector{Vector{Nothing}}()
id(F::FunctionOperator)      = "$(nameof(F.f))"
(F::FunctionOperator)(::Vector{<:AbstractArray}) = F

function (F::FunctionOperator{D,R,L})(x::V) where {D<:Number,R<:Number,L<:Linearity,V<:AbstractVector{D}}
    return R.(F.f(x))
end