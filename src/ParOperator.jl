export Linearity, Linear, NonLinear
export Parametricity, Parametric, NonParametric, Parameterized
export ASTLocation, Internal, External
export Applicable
export promote_linearity, promote_parametricity
export ParOperator
export ParLinearOperator, ParNonLinearOperator
export ParParametricOperator, ParNonParametricOperator, ParParameterizedOperator, ParApplicableOperator
export ParInternalOperator, ParExternalOperator
export ParAdjoint, ParParameterized

@typeflag Linearity     Linear     NonLinear
@typeflag Parametricity Parametric NonParametric Parameterized
@typeflag ASTLocation   Internal   External

const Applicable = Union{NonParametric, Parameterized}

promote_linearity(::Type{Linear}, ::Type{Linear})           = Linear
promote_linearity(::Type{<:Linearity}, ::Type{<:Linearity}) = NonLinear

promote_parametricity(::Type{NonParametric}, ::Type{NonParametric}) = NonParametric
promote_parametricity(::Type{Parametric}, ::Type{Parametric})       = Parametric
promote_parametricity(::Type{<:Applicable}, ::Type{Parametric})     = Parametric
promote_parametricity(::Type{Parametric}, ::Type{<:Applicable})     = Parametric
promote_parametricity(::Type{Parameterized}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{NonParametric}) = Parameterized
promote_parametricity(::Type{NonParametric}, ::Type{Parameterized}) = Parameterized

abstract type ParOperator{
    D<:Number,
    R<:Number,
    L<:Linearity,
    P<:Parametricity,
    T<:ASTLocation
} end

const ParLinearOperator{D,R,P,T}        = ParOperator{D,R,Linear,P,T}
const ParNonLinearOperator{D,R,P,T}     = ParOperator{D,R,NonLinear,P,T}
const ParParametricOperator{D,R,L,T}    = ParOperator{D,R,L,Parametric,T}
const ParNonParametricOperator{D,R,L,T} = ParOperator{D,R,L,NonParametric,T}
const ParParameterizedOperator{D,R,L,T} = ParOperator{D,R,L,Parameterized,T}
const ParApplicableOperator{D,R,L,T}    = ParOperator{D,R,L,<:Applicable,T}
const ParInternalOperator{D,R,L,P}      = ParOperator{D,R,L,P,Internal}
const ParExternalOperator{D,R,L,P}      = ParOperator{D,R,L,P,External}


struct ParAdjoint{D,R,P,F} <: ParLinearOperator{R,D,P,Internal}
    op::F
    function ParAdjoint(F::ParLinearOperator{D,R,P,T}) where {D,R,P,T}
        return new{D,R,P,typeof(F)}(F)
    end
end

struct ParParameterized{D,R,L,F,V} <: ParOperator{D,R,L,Parameterized,Internal}
    op::F
    θ::V
    function ParParameterized(F::ParParametricOperator{D,R,L,T}, θ::AbstractVector{<:Number}) where {D,R,L,T}
        return new{D,R,L,typeof(F),typeof(θ)}(F, θ)
    end
end