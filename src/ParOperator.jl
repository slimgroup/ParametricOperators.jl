export ParOperator, ParLinearOperator
export Linearity, Linear, NonLinear
export Parametricity, Parametric, NonParametric, Parameterized, Applicable
export Order, FirstOrder, HigherOrder

abstract type Linearity end
struct Linear <: Linearity end
struct NonLinear <: Linearity end

abstract type Parametricity end
struct Parametric <: Parametricity end
struct NonParametric <: Parametricity end
struct Parameterized <: Parametricity end

const Applicable = Union{NonParametric, Parameterized}

abstract type Order end
struct FirstOrder <: Order end
struct HigherOrder <: Order end

abstract type ParOperator{
    D <: Option{Number},
    R <: Option{Number},
    L <: Linearity,
    P <: Parametricity,
    O <: Order
} end

const ParLinearOperator{D,R,P,O} = ParOperator{D,R,Linear,P,O}