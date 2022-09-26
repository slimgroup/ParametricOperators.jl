export TaoBias

struct TaoBias{D,R} <: TaoOperator{D,R,NonLinear,Parametric,External}
    n::Int64
    id::Any
end

TaoBias{T}(n::Int64) where {T} = TaoBias{T,T}(n, uuid4(GLOBAL_RNG))
TaoBias{D,R}(n::Int64) where {D,R} = TaoBias{D,R}(n, uuid4(GLOBAL_RNG))
TaoBias(F::TaoOperator{D,R,<:Linearity,<:Parametricity,<:NodeType}) where {D,R} =
    TaoBias{D,R}(Range(F))

Domain(::TaoBias) = nothing
Range(B::TaoBias) = B.n
id(B::TaoBias) = B.id
nparams(B::TaoBias) = B.n
init(B::TaoBias{D,R}) where {D,R} = zeros(R, B.n)

(B::TaoParameterized{D,R,NonLinear,TaoBias{D,R},T,V})(::X) where
    {D,R,X<:AbstractVector{D},T,V<:AbstractVector{T}} = B.θ