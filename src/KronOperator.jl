struct LinearKron{D,R1,R2} <: AbstractLinearOperator{D,R2}
    A::AbstractLinearOperator{R1,R2}
    B::AbstractLinearOperator{D,R1}
    row_first::Bool
end

function ⊗(A::AbstractLinearOperator{R1,R2}, B::AbstractLinearOperator{D,R1}) where {D,R1,R2}
    return LinearKron{D,R1,R2}(A, B, true)
end

Domain(L::LinearKron) = Domain(L.A)*Domain(L.B)
Range(L::LinearKron) = Range(L.A)*Range(L.B)
param(L::LinearKron) = [param(L.A)..., param(L.B)...]
nparam(L::LinearKron) = nparam(L.A) + nparam(L.B) 

function init(L::LinearKron, pv::Optional{ParameterVector} = nothing)
    θA = init(L.A, pv)
    θB = init(L.B, pv)
    θL = [θA..., θB...]
    if !isnothing(pv)
        pv[id(L)] = θL
    end
    return θL
end

adjoint(L::LinearKron{D,R1,R2}) where {D,R1,R2} =
    LinearKron{R2,R1,D}(adjoint(L.A), adjoint(L.B), !L.row_first)
id(L::LinearKron) = "$(id(L.A))_kron_$(id(L.B))"

function *(L::LinearKron{D,R1,R2}, x::AbstractVector{D}) where {D,R1,R2}
    shape = (Domain(L.B), Domain(L.A))
    y = reshape(x, shape)
    if L.row_first
        y = mapslices(r -> L.B*r, y, dims = [1])
        y = mapslices(c -> L.A*c, y, dims = [2])
    else
        y = mapslices(c -> L.A*c, y, dims = [2])
        y = mapslices(r -> L.B*r, y, dims = [1])
    end
    return vec(y)
end

function *(L::LinearKron{D,R1,R2}, x::AbstractVecOrMat{D}) where {D,R1,R2}
    nv = size(x)[2]
    shape = (Domain(L.B), Domain(L.A), nv)
    y = reshape(x, shape)
    if L.row_first
        y = mapslices(r -> L.B*r, y, dims = [1,3])
        y = mapslices(c -> L.A*c, y, dims = [2,3])
    else
        y = mapslices(c -> L.A*c, y, dims = [2,3])
        y = mapslices(r -> L.B*r, y, dims = [1,3])
    end
    return reshape(y, (Range(L), nv))
end

(L::LinearKron)(θs::Any...) = LinearKron(
    L.A(select(1, nparam(L.A), collect(θs))...),
    L.B(select(nparam(L.A)+1, nparam(L.A)+nparam(L.B), collect(θs))...),
    L.row_first
)
(L::LinearKron)(pv::ParameterVector) = LinearKron(L.A(pv), L.B(pv), L.row_first)
