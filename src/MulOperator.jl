struct LinearMul{D,R1,R2} <: AbstractLinearOperator{D,R2}
    A::AbstractLinearOperator{R1,R2}
    B::AbstractLinearOperator{D,R1}
end

function *(A::AbstractLinearOperator{R1,R2}, B::AbstractLinearOperator{D,R1}) where {D,R1,R2}
    @assert Range(B) == Domain(A)
    return LinearMul{D,R1,R2}(A, B)
end

Domain(L::LinearMul) = Domain(L.B)
Range(L::LinearMul) = Range(L.A)
param(L::LinearMul) = [param(L.A)..., param(L.B)...]
nparam(L::LinearMul) = nparam(L.A) + nparam(L.B) 

function init(L::LinearMul, pv::Optional{ParameterVector} = nothing)
    θA = init(L.A, pv)
    θB = init(L.B, pv)
    θL = [θA..., θB...]
    if !isnothing(pv)
        pv[id(L)] = θL
    end
    return θL
end

adjoint(L::LinearMul) = LinearMul(adjoint(L.B), adjoint(L.A))
id(L::LinearMul) = "$(id(L.A))_mul_$(id(L.B))"
*(L::LinearMul{D,R}, x::AbstractVector{D}) where {D,R} = L.A*(L.B*x)
*(L::LinearMul{D,R}, x::AbstractVecOrMat{D}) where {D,R} = L.A*(L.B*x)

(L::LinearMul)(θs::Any...) = LinearMul(
    L.A(select(1, nparam(L.A), collect(θs))...),
    L.B(select(nparam(L.A)+1, nparam(L.A)+nparam(L.B), collect(θs))...)
)
(L::LinearMul)(pv::ParameterVector) = LinearMul(L.A(pv), L.B(pv))
