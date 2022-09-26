export TaoOperator, TaoAdjoint, TaoParameterized
export Linearity, Linear, NonLinear
export Parametricity, Parametric, NonParametric
export Distribution, Distributed, NonDistributed

@typeflag Linearity     Linear      NonLinear
@typeflag Parametricity Parametric  NonParametric Parameterized
@typeflag NodeType      Internal    External

const NoAcceptParams = Union{NonParametric, Parameterized}

abstract type TaoOperator{
    D<:Number,
    R<:Number,
    L<:Linearity,
    P<:Parametricity,
    T<:NodeType
} end

struct TaoAdjoint{
    D<:Number,
    R<:Number,
    P<:Parametricity,
    F<:TaoOperator{D,R,Linear,P,<:NodeType}
} <: TaoOperator{R,D,Linear,P,Internal}
    op::F
end

struct TaoParameterized{
    D<:Number,
    R<:Number,
    L<:Linearity,
    F<:TaoOperator{D,R,L,Parametric,External},
    T<:Number,
    V<:AbstractVector{T}
} <: TaoOperator{D,R,L,Parameterized,Internal}
    op::F
    θ::V
end