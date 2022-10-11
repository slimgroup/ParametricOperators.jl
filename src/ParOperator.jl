export Linearity, Linear, NonLinear
export Parametricity, Parametric, NonParametric, Parameterized, Applicable
export ASTLocation, Internal, External
export promote_optype, promote_opdim, promote_linearity, promote_parametricity
export ParOperator

abstract type Linearity end
struct Linear <: Linearity end
struct NonLinear <: Linearity end

abstract type Parametricity end
struct Parametric <: Parametricity end
struct NonParametric <: Parametricity end
struct Parameterized <: Parametricity end

const Applicable = Union{NonParametric, Parameterized}

abstract type ASTLocation end
struct Internal <: ASTLocation end
struct External <: ASTLocation end

promote_linearity(::Type{Linear}, ::Type{Linear}) = Linear
promote_linearity(::Type{<:Linearity}, ::Type{<:Linearity}) = NonLinear

promote_parametricity(::Type{NonParametric}, ::Type{NonParametric}) = NonParametric
promote_parametricity(::Type{Parameterized}, ::Type{NonParametric}) = Parameterized
promote_parametricity(::Type{NonParametric}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{Parameterized}, ::Type{Parameterized}) = Parameterized
promote_parametricity(::Type{<:Parametricity}, ::Type{<:Parametricity}) = Parametric

abstract type ParOperator{
    D<:Optional{<:Number},
    R<:Optional{<:Number},
    L<:Linearity,
    P<:Parametricity,
    T<:ASTLocation
} end

const ParLinearOperator{D,R,P,T} = ParOperator{D,R,Linear,P,T}