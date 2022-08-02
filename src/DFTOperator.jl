struct DFTOperator{T<:Complex} <: AbstractLinearOperator{T,T}
    shape::AbstractVector{<:Integer}
    adjoint::Bool
    id::Any
end

function DFTOperator{T}(shape::AbstractVector{<:Integer}) where {T<:Complex}
    return DFTOperator{T}(
        shape,
        false,
        uuid4(GLOBAL_RNG)
    )
end

Domain(F::DFTOperator) = prod(F.shape)
Range(F::DFTOperator) = prod(F.shape)
param(F::DFTOperator) = []
nparam(F::DFTOperator) = 0

function init(F::DFTOperator, pv::Optional{ParameterVector})
    if !isnothing(pv)
        pv[F.id] = []
    end
    return []
end

adjoint(F::DFTOperator{T}) where{T} =
    DFTOperator{T}(
        F.shape,
        !F.adjoint,
        F.id
    )

id(F::DFTOperator) = F.id

function *(F::DFTOperator{T}, x::AbstractVector{T}) where {T}
    if F.adjoint
        rx = reshape(x, tuple(F.shape...))
        ry = ifft(rx)*T(sqrt(Range(F)))
        y = vec(ry)
        return y
    else
        rx = reshape(x, tuple(F.shape...))
        ry = fft(rx)/T(sqrt(Domain(F)))
        y = vec(ry)
        return y
    end
end

(F::DFTOperator{T})(θs::Any...) where {T} = F
(F::DFTOperator{T})(pv::ParameterVector) where {T} = F
