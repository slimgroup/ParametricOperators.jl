struct DRFTOperator{D<:Union{Real,Complex},R<:Complex} <: AbstractLinearOperator{D,R}
    shape::AbstractVector{<:Integer}
    adjoint::Bool
    id::Any
end

function DRFTOperator{D,R}(shape::AbstractVector{<:Integer}) where {D,R}
    return DRFTOperator{D,R}(
        shape,
        false,
        uuid4(GLOBAL_RNG)
    )
end

Domain(F::DRFTOperator) = F.adjoint ? prod([(F.shape[1]÷2)+1, F.shape[2:end]...]) : prod(F.shape)
Range(F::DRFTOperator) = F.adjoint ? prod(F.shape) : prod([(F.shape[1]÷2)+1, F.shape[2:end]...])
param(F::DRFTOperator) = [F.shape]
nparam(F::DRFTOperator) = 1

function init(F::DRFTOperator, pv::Optional{ParameterVector})
    θ = F.shape
    if !isnothing(pv)
        pv[F.id] = [θ]
    end
    return [θ]
end

adjoint(F::DRFTOperator{D,R}) where{D,R} =
    DRFTOperator{D,R}(
        F.shape,
        !F.adjoint,
        F.id
    )

id(F::DRFTOperator) = F.id

function *(F::DRFTOperator{D,R}, x::AbstractVector{T}) where {D,R,T<:Union{D,R}}
    shape_out = ((F.shape[1]÷2)+1, F.shape[2:end]...)
    if F.adjoint
        @assert eltype(x) == R
        rx = zeros(T, F.shape...)
        rx[[1:s for s in shape_out]...] .= reshape(x, shape_out)
        ry = ifft(rx)*T(sqrt(prod(F.shape)))
        y = vec(ry)
        return y
    else
        @assert eltype(x) == D
        rx = reshape(x, tuple(F.shape...))
        ry = fft(rx)/T(sqrt(prod(F.shape)))
        y = vec(ry[1:(shape_out[1]÷2+1), [1:s for s in shape_out[2:end]]...])
        return y
    end
end

(F::DRFTOperator{D,R})(θ::AbstractVector{<:Integer}) where {D,R} =
    DRFTOperator{D,R}(
        θ,
        F.adjoint,
        F.id
    )

(F::DRFTOperator{D,R})(pv::ParameterVector) where {D,R} =
    DRFTOperator{D,R}(
        pv[F.id][1],
        F.adjoint,
        F.id
    )
