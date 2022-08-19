import Base.kron

struct KronOperator{D1,D2,R,LHS<:AbstractLinearOperator{D1,R},RHS<:AbstractLinearOperator{D2,R}} <: AbstractLinearOperator{Union{D1,D2},R}
    lhs::LHS
    rhs::RHS
    id::Any
end

function kron(lhs::LHS, rhs::RHS) where
    {
        D1,
        D2,
        R,
        LHS<:AbstractLinearOperator{D1,R},
        RHS<:AbstractLinearOperator{D2,R},
    }
    return KronOperator{D1,D2,R,LHS,RHS}(
        lhs,
        rhs,
        uid()
    )
end

function ⊗(lhs::LHS, rhs::RHS) where 
    {
        D1,
        D2,
        R,
        LHS<:AbstractLinearOperator{D1,R},
        RHS<:AbstractLinearOperator{D2,R},
    }
    return kron(lhs, rhs)
end

ddt(::KronOperator{D,D,R,LHS,RHS}) where {D,R,LHS<:AbstractLinearOperator{D,R},RHS<:AbstractLinearOperator{D,R}} = D
function ddt(::KronOperator{D1,D2,R,LHS,RHS}) where {D1,D2,R,LHS<:AbstractLinearOperator{D1,R},RHS<:AbstractLinearOperator{D2,R}}
    return issubsettypeof(D1, D2) ? D1 : D2
end
Domain(K::KronOperator) = Domain(K.lhs)*Domain(K.rhs)
Range(K::KronOperator) = Range(K.lhs)*Range(K.rhs)
params(K::KronOperator) = [params(K.lhs)..., params(K.rhs)...]
nparams(K::KronOperator) = nparams(K.lhs) + nparams(K.rhs)
init(K::KronOperator, pc::Optional{ParameterContainer} = nothing) = [init(K.lhs, pc)..., init(K.rhs, pc)...]
id(K::KronOperator) = K.id

function (K::KronOperator{D1,D2,R,LHS,RHS})(θs::Vararg{<:AbstractArray}) where
    {
        D1,
        D2,
        R,
        LHS<:AbstractLinearOperator{D1,R},
        RHS<:AbstractLinearOperator{D2,R}
    }
    return KronOperator{D1,D2,R,LHS,RHS}(
        K.lhs(θs[1:nparams(K.lhs)]...),
        K.rhs(θs[nparams(K.lhs)+1:end]...),
        K.id
    )
end

function *(K::KronOperator{D,D,R,LHS,RHS}, x::V) where
    {
        D,
        R,
        LHS<:AbstractLinearOperator{D,R},
        RHS<:AbstractLinearOperator{D,R},
        V<:AbstractVector{D}
    }
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = mapreduce(c -> K.rhs*c, hcat, xr)
    y2 = mapreduce(r -> K.lhs*vec(r), vcat, y1)
    return vec(y2)
end

function *(A::LinearOperatorAdjoint{D,D,KronOperator{D,D,R,LHS,RHS}}, y::V) where
    {
        D,
        R,
        LHS<:AbstractLinearOperator{D,R},
        RHS<:AbstractLinearOperator{D,R},
        V<:AbstractVector{R}
    }
    y2 = reshape(y, Range(A.inner.rhs), Range(A.inner.lhs))
    y1 = mapreduce(r -> A.inner.lhs'*vec(r), vcat, y2)
    xr = mapreduce(c -> A.inner.rhs'*c, hcat, y1)
    return vec(xr)
end

function *(K::KronOperator{D1,D2,R,LHS,RHS}, x::V) where
    {
        D1,
        D2,
        R,
        LHS<:AbstractLinearOperator{D1,R},
        RHS<:AbstractLinearOperator{D2,R},
        V<:AbstractVector{<:Union{D1,D2}}
    }
    if issubsettypeof(D1, D2)
        xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
        y1 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(xr))
        yt = D2.(y1)
        y2 = mapreduce(c -> K.rhs*c, hcat, eachcol(yt))
        return vec(y2)
    else
        xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
        y1 = mapreduce(c -> K.rhs*c, hcat, eachcol(xr))
        yt = D1.(y1)
        y2 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(yt))
        return vec(y2)
    end
end

function *(A::LinearOperatorAdjoint{Union{D1,D2},R,K}, y::V) where
    {
        D1,
        D2,
        R,
        LHS<:AbstractLinearOperator{D1,R},
        RHS<:AbstractLinearOperator{D2,R},
        K<:KronOperator{D1,D2,R,LHS,RHS},
        V<:AbstractVector{<:Union{D1,D2}}
    }
    if issubsettypeof(D1, D2)
        y2 = reshape(y, Range(A.inner.rhs), Range(A.inner.lhs))
        yt = mapreduce(c -> A.inner.rhs'*c, hcat, eachcol(y2))
        y1 = D1.(yt)
        xr = mapreduce(r -> transpose(A.inner.lhs'*r), vcat, eachrow(y1))
        return vec(xr)
    else
        y2 = reshape(y, Range(A.inner.rhs), Range(A.inner.lhs))
        yt = mapreduce(r -> transpose(A.inner.lhs'*r), vcat, eachrow(y2))
        y1 = D2.(yt)
        xr = mapreduce(c -> A.inner.rhs'*c, hcat, eachcol(y1))
        return vec(xr)
    end
end