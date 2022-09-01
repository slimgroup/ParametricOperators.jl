struct BiasOperator{D,R} <: Operator{D,R,NonLinear,NonParametric}
    m::Int64
    n::Int64
    id::Any
end

BiasOperator{D,R}(m::Int64, n::Int64) where {D,R} =
    BiasOperator{D,R}(
        m,
        n,
        uid()
    )

BiasOperator(F::Operator{D,R,<:Linearity,<:Parametricity}) where {D,R} =
    BiasOperator{D,R}(
        Range(F),
        Domain(F),
        uid()
    )

Domain(b::BiasOperator) = b.n
Range(b::BiasOperator)  = b.m
nparams(::BiasOperator) = 1
init(b::BiasOperator{D,R}) where {D,R} = [zeros(R, b.m)]
id(b::BiasOperator) = b.id

(b::Parameterized{D,R,NonLinear,BiasOperator{D,R}})(::AbstractVector{D}) where {D<:Number,R<:Number} = b.θ[1]