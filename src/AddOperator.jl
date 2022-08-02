struct LinearAdd{D,R} <: AbstractLinearOperator{D,R}
    A::AbstractLinearOperator{D,R}
    B::AbstractLinearOperator{D,R}
end

function +(A::AbstractLinearOperator{D,R}, B::AbstractLinearOperator{D,R}) where {D,R}
    @assert Domain(A) == Domain(B)
    @assert Range(A) == Range(B)
    return LinearAdd{D,R}(A, B)
end

Domain(L::LinearAdd) = Domain(L.A)
Range(L::LinearAdd) = Range(L.A)
param(L::LinearAdd) = [param(L.A)..., param(L.B)...]
nparam(L::LinearAdd) = nparam(L.A) + nparam(L.B) 

function init(L::LinearAdd, pv::Optional{ParameterVector} = nothing)
    θA = init(L.A, pv)
    θB = init(L.B, pv)
    θL = [θA..., θB...]
    if !isnothing(pv)
        pv[id(L)] = θL
    end
    return θL
end

adjoint(L::LinearAdd) = LinearAdd(adjoint(L.A), adjoint(L.B))
id(L::LinearAdd) = "$(id(L.A))_plus_$(id(L.B))"
*(L::LinearAdd{D,R}, x::AbstractVector{D}) where {D,R} = L.A*x + L.B*x
*(L::LinearAdd{D,R}, x::AbstractVecOrMat{D}) where {D,R} = L.A*x + L.B*x

(L::LinearAdd)(θs::Any...) = LinearAdd(
    L.A(select(1, nparam(L.A), collect(θs))...),
    L.B(select(nparam(L.A)+1, nparam(L.A)+nparam(L.B), collect(θs))...)
)
(L::LinearAdd)(pv::ParameterVector) = LinearAdd(L.A(pv), L.B(pv))
