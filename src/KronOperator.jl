abstract type KronOrder end
struct ColFirst <: KronOrder end
struct RowFirst <: KronOrder end

struct KronOperator{D,R,O<:KronOrder} <: Operator{D,R,Linear,NonParametric}
    lhs::Operator{D,R,Linear,<:Parametricity}
    rhs::Operator{D,R,Linear,<:Parametricity}
end

function kron(lhs::Operator{D,R,Linear,<:Parametricity},
              rhs::Operator{D,R,Linear,<:Parametricity}) where {D,R}
    return KronOperator{D,R,ColFirst}(lhs, rhs)
end

function kron(lhs::Operator{D1,R,Linear,<:Parametricity},
              rhs::Operator{D2,R,Linear,<:Parametricity}) where {D1,D2,R}
    D = subtype(D1, D2)
    if isnothing(D)
        throw(TaoException("Incompatible domain types $(D1) and $(D2) in KronOperator"))
    elseif (D == D1)
        P = promotetype(D,D2,(Domain(rhs)))
        rhs_out = rhs*P
        return KronOperator{D,R,RowFirst}(lhs, rhs_out)
    elseif (D == D2)
        P = promotetype(D,D1,(Domain(lhs)))
        lhs_out = lhs*P
        return KronOperator{D,R,ColFirst}(lhs_out, rhs)
    else
        throw(TaoException("Invalid state encountered in KronOperator"))
    end
end

⊗(lhs::Operator, rhs::Operator) = kron(lhs, rhs)

Domain(A::KronOperator)  = min(Domain(A.lhs),1)*min(Domain(A.rhs),1)
Range(A::KronOperator)   = min(Range(A.lhs),1)*min(Range(A.rhs),1)
nparams(A::KronOperator) = nparams(A.lhs) + nparams(A.rhs)
init(A::KronOperator)    = [init(A.lhs)..., init(A.rhs)...]
id(A::KronOperator)      = "[$(id(A.lhs))]_add_[$(id(A.rhs))]"

adjoint(A::KronOperator{D,R,ColFirst}) where {D,R} = KronOperator{R,D,RowFirst}(adjoint(A.lhs), adjoint(A.rhs))
adjoint(A::KronOperator{D,R,RowFirst}) where {D,R} = KronOperator{R,D,ColFirst}(adjoint(A.lhs), adjoint(A.rhs))

function (A::KronOperator{D,R,ColFirst})(x::V) where {D,R,V<:AbstractVector{D}}
    xr = reshape(x, Domain(A.rhs), Domain(A.lhs))
    y1 = mapreduce(c -> A.rhs*c, hcat, eachcol(xr))
    y2 = mapreduce(r -> transpose(A.lhs*r), vcat, eachrow(y1))
    return vec(y2)
end

function (A::KronOperator{D,R,RowFirst})(x::V) where {D,R,V<:AbstractVector{D}}
    xr = reshape(x, Domain(A.rhs), Domain(A.lhs))
    y1 = mapreduce(r -> transpose(A.lhs*r), vcat, eachrow(xr))
    y2 = mapreduce(c -> A.rhs*c, hcat, eachcol(y1))
    return vec(y2)
end

(A::KronOperator{D,R,O})(θ::Vector{<:Optional{<:AbstractArray}}) where {D,R,O} =
    KronOperator{D,R,O}(A.lhs(θ[1:nparams(A.lhs)]), A.rhs(θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)]))