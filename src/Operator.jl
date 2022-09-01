abstract type Linearity end
struct Linear <: Linearity end
struct NonLinear <: Linearity end

abstract type Parametricity end
struct Parametric <: Parametricity end
struct NonParametric <: Parametricity end

abstract type ParamOrdering end
struct LeftFirst <: ParamOrdering end
struct RightFirst <: ParamOrdering end

abstract type Operator{D<:Number,R<:Number,L<:Linearity,P<:Parametricity} end

struct Parameterized{D,R,L,F<:Operator{D,R,L,NonParametric}} <: Operator{D,R,L,Parametric}
    op::F
    θ::Vector{<:AbstractArray}
end

struct Adjoint{D,R,P,F<:Operator{D,R,Linear,P}} <: Operator{R,D,Linear,P}
    op::F
end

Domain(::Operator)  = throw(TaoException(""))
Range(::Operator)   = throw(TaoException(""))
nparams(::Operator) = throw(TaoException(""))
init(::Operator)    = throw(TaoException(""))
id(::Operator)      = throw(TaoException(""))

Domain(F::Parameterized)  = Domain(F.op)
Range(F::Parameterized)   = Range(F.op)
nparams(F::Parameterized) = length(F.θ)
init(F::Parameterized)    = F.θ
id(F::Parameterized)      = "parameterized_[$(id(F.op))]"

Domain(A::Adjoint)  = Range(A.op)
Range(A::Adjoint)   = Domain(A.op)
nparams(A::Adjoint) = nparams(A.op)
init(A::Adjoint)    = init(A.op)
id(A::Adjoint)      = "adjoint_[$(id(A.op))]"

DDT(F::Operator{D,R,<:Linearity,<:Parametricity}) where {D,R} = D
RDT(F::Operator{D,R,<:Linearity,<:Parametricity}) where {D,R} = R

(::Operator{D,R,L,P})(::AbstractVector{D}) where {D,R,L,P} = throw(TaoException(""))
*(A::Operator{D,R,Linear,P}, x::AbstractVector{D}) where {D,R,P} = A(x)
(F::Operator{D,R,L,P})(x::AbstractVecOrMat{D}) where {D,R,L,P} =
    mapreduce(c -> F(c), hcat, eachcol(x))
(F::Operator{D,R,L,NonParametric})(θ::Vector{<:AbstractArray}) where {D,R,L} =
    Parameterized(F, θ)
(F::Operator{D,R,L,Parametric})(::Vector{<:AbstractArray}) where {D,R,L} = F
(F::Operator{D,R,L,NonParametric})(x::AbstractVector{D}, θ::Vector{<:AbstractArray}) where {D,R,L} = F(θ)(x)

adjoint(A::Operator{D,R,Linear,P}) where {D,R,P} = Adjoint(A)
adjoint(A::Parameterized{D,R,Linear,F}) where {D,R,F<:Operator{D,R,Linear,NonParametric}} = Parameterized(adjoint(A.op), A.θ)