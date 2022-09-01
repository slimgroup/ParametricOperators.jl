struct PromoteOperator{D,R} <: Operator{D,R,Linear,NonParametric}
    n::Int64
    id::Any
end

promotetype(D, R, n) = PromoteOperator{D,R}(n, uid())

Domain(P::PromoteOperator)  = P.n
Range(P::PromoteOperator)   = P.n
nparams(P::PromoteOperator) = 0
init(::PromoteOperator)     = Vector{Vector{Nothing}}()
id(P::PromoteOperator)      = P.id

(P::PromoteOperator)(::Vector{<:AbstractArray}) = P
(P::PromoteOperator{D,R})(x::V) where {D<:Number,R<:Number,V<:AbstractVector{D}} = R.(x)
(P::Adjoint{D,R,NonParametric,PromoteOperator{D,R}})(x::V) where {D<:Number,R<:Number,V<:AbstractVector{R}} = D.(x)