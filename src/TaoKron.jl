export TaoKron, ⊗

struct TaoKron{
    D<:Number,
    D1<:Number,
    D2<:Number,
    R<:Number,
    R1<:Number,
    R2<:Number,
    P<:Parametricity,
    P1<:Parametricity,
    P2<:Parametricity,
    F1<:TaoOperator{D1,R1,Linear,P1,<:NodeType},
    F2<:TaoOperator{D2,R2,Linear,P2,<:NodeType}
} <: TaoOperator{D,R,Linear,P,Internal}
    lhs::F1
    rhs::F2
end

function kron(lhs::F1, rhs::F2) where
    {
        D<:Number,
        T<:Number,
        R<:Number,
        P1<:Parametricity,
        P2<:Parametricity,
        F1<:TaoOperator{T,R,Linear,P1,<:NodeType},
        F2<:TaoOperator{D,T,Linear,P2,<:NodeType}
    }
    P = @match (P1, P2) begin
        (_::Type{NonParametric}, _::Type{NonParametric}) => NonParametric
        (_::Type{Parameterized}, _::Type{Parameterized}) => Parameterized
        (_::Type{NonParametric}, _::Type{Parameterized}) => Parameterized
        (_::Type{Parameterized}, _::Type{NonParametric}) => Parameterized
        _ => Parametric
    end
    return TaoKron{D,T,D,R,R,T,P,P1,P2,F1,F2}(lhs, rhs)
end

⊗(lhs::TaoOperator, rhs::TaoOperator) = kron(lhs, rhs)

Domain(K::TaoKron) = Domain(K.lhs)*Domain(K.rhs)
Range(K::TaoKron) = Range(K.lhs)*Range(K.rhs)
children(K::TaoKron) = [K.lhs, K.rhs]
id(K::TaoKron) = "[$(id(K.lhs))]_kron_[$(id(K.rhs))]"

init(K::TaoKron{D,D1,D2,R,R1,R2,Parametric,Parametric,Parametric,F1,F2}) where 
    {D,D1,D2,R,R1,R2,F1,F2} = mapreduce(init, vcat, children(K))

init(K::TaoKron{D,D1,D2,R,R1,R2,Parametric,Parametric,<:NoAcceptParams,F1,F2}) where 
    {D,D1,D2,R,R1,R2,F1,F2} = init(K.lhs)

init(K::TaoKron{D,D1,D2,R,R1,R2,Parametric,<:NoAcceptParams,Parametric,F1,F2}) where 
    {D,D1,D2,R,R1,R2,F1,F2} = init(K.rhs)

function adjoint(K::TaoKron{D,D1,D2,R,R1,R2,P,P1,P2,F1,F2}) where
        {D,D1,D2,R,R1,R2,P,P1,P2,F1,F2}
    lhs_out = adjoint(K.lhs)
    rhs_out = adjoint(K.rhs)
    return TaoKron{R,R1,R2,D,D1,D2,P,P1,P2,typeof(lhs_out), typeof(rhs_out)}(lhs_out, rhs_out)
end

function (K::TaoKron{D,D1,D2,R,R1,R2,Parametric,Parametric,Parametric,F1,F2})(θ::V) where
    {D,D1,D2,R,R1,R2,F1,F2,V<:AbstractVector}
    lhs_out = K.lhs(@view θ[1:nparams(K.lhs)])
    rhs_out = K.rhs(@view θ[nparams(K.lhs)+1:nparams(K.lhs)+nparams(K.rhs)])
    return TaoKron{D,D1,D2,R,R1,R2,Parameterized,Parameterized,Parameterized,typeof(lhs_out),typeof(rhs_out)}(lhs_out, rhs_out)
end

function (K::TaoKron{D,D1,D2,R,R1,R2,Parametric,Parametric,P,F1,F2})(θ::V) where
    {D,D1,D2,R,R1,R2,P<:NoAcceptParams,F1,F2,V<:AbstractVector}
    lhs_out = K.lhs(θ)
    return TaoKron{D,D1,D2,R,R1,R2,Parameterized,Parameterized,P,typeof(lhs_out),F2}(lhs_out, K.rhs)
end

function (K::TaoKron{D,D1,D2,R,R1,R2,Parametric,P,Parametric,F1,F2})(θ::V) where
    {D,D1,D2,R,R1,R2,P<:NoAcceptParams,F1,F2,V<:AbstractVector}
    rhs_out = K.rhs(θ)
    return TaoKron{D,D1,D2,R,R1,R2,Parameterized,P,Parameterized,F1,typeof(rhs_out)}(K.lhs, rhs_out)
end

function (K::TaoKron{T,T,T,T,T,T,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,F2})(x::X) where
    {T,F1,F2,X<:AbstractVector{T}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = mapreduce(c -> K.rhs*c, hcat, eachcol(xr))
    y2 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(y1))
    return vec(y2)
end

function (K::TaoKron{D,D,T,R,T,R,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,F2})(x::X) where
    {D,T,R,F1,F2,X<:AbstractVector{D}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = mapreduce(c -> K.rhs*c, hcat, eachcol(xr))
    y2 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(y1))
    return vec(y2)
end

function (K::TaoKron{D,T,D,R,R,T,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,F2})(x::X) where
    {D,T,R,F1,F2,X<:AbstractVector{D}}
    xr = reshape(x, Domain(K.rhs), Domain(K.lhs))
    y1 = mapreduce(r -> transpose(K.lhs*r), vcat, eachrow(xr))
    y2 = mapreduce(c -> K.rhs*c, hcat, eachcol(y1))
    return vec(y2)
end