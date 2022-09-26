export TaoCompose

@typeflag ParamOrder LeftFirst RightFirst

struct TaoCompose{
    D<:Number,
    T<:Number,
    R<:Number,
    L<:Linearity,
    L1<:Linearity,
    L2<:Linearity,
    P<:Parametricity,
    P1<:Parametricity,
    P2<:Parametricity,
    F1<:TaoOperator{T,R,L1,P1,<:NodeType},
    F2<:TaoOperator{D,T,L2,P2,<:NodeType},
    O<:ParamOrder
} <: TaoOperator{D,R,L,P,Internal}
    lhs::F1
    rhs::F2
end

function ∘(lhs::F1, rhs::F2) where
    {
        D<:Number,
        T<:Number,
        R<:Number,
        L1<:Linearity,
        L2<:Linearity,
        P1<:Parametricity,
        P2<:Parametricity,
        F1<:TaoOperator{T,R,L1,P1,<:NodeType},
        F2<:TaoOperator{D,T,L2,P2,<:NodeType},
    }
    @assert dims_compatible(Domain(lhs), Range(rhs))
    L = @match (L1, L2) begin
        (_::Type{Linear}, _::Type{Linear}) => Linear
        _ => NonLinear
    end
    P = @match (P1, P2) begin
        (_::Type{NonParametric}, _::Type{NonParametric}) => NonParametric
        (_::Type{Parameterized}, _::Type{Parameterized}) => Parameterized
        (_::Type{NonParametric}, _::Type{Parameterized}) => Parameterized
        (_::Type{Parameterized}, _::Type{NonParametric}) => Parameterized
        _ => Parametric
    end
    return TaoCompose{D,T,R,L,L1,L2,P,P1,P2,F1,F2,LeftFirst}(lhs, rhs)
end

*(lhs::TaoOperator{T,R,Linear,<:Parametricity,<:NodeType}, rhs::TaoOperator{D,T,Linear,<:Parametricity,<:NodeType}) where
    {D,T,R} = lhs ∘ rhs

function adjoint(C::TaoCompose{D,T,R,Linear,Linear,Linear,P,P1,P2,F1,F2,RightFirst}) where
    {
        D,T,R,P,P1,P2,F1,F2
    }
    lhs_out = adjoint(C.rhs)
    rhs_out = adjoint(C.lhs)
    return TaoCompose{R,T,D,Linear,Linear,Linear,P,P2,P1,typeof(lhs_out),typeof(rhs_out),LeftFirst}(lhs_out, rhs_out)
end

function adjoint(C::TaoCompose{D,T,R,Linear,Linear,Linear,P,P1,P2,F1,F2,LeftFirst}) where
    {
        D,T,R,P,P1,P2,F1,F2
    }
    lhs_out = adjoint(C.rhs)
    rhs_out = adjoint(C.lhs)
    return TaoCompose{R,T,D,Linear,Linear,Linear,P,P2,P1,typeof(lhs_out),typeof(rhs_out),RightFirst}(lhs_out, rhs_out)
end

Domain(C::TaoCompose) = Domain(C.rhs)
Range(C::TaoCompose) = Range(C.lhs)
children(C::TaoCompose) = [C.lhs, C.rhs]
id(C::TaoCompose) = "[$(id(C.lhs))]_comp_[$(id(C.rhs))]"

init(C::TaoCompose{D,T,R,L,L1,L2,Parametric,Parametric,Parametric,F1,F2}) where {D,T,R,L,L1,L2,F1,F2} = mapreduce(init, vcat, children(C))
init(C::TaoCompose{D,T,R,L,L1,L2,Parametric,Parametric,<:NoAcceptParams,F1,F2}) where {D,T,R,L,L1,L2,F1,F2} = init(C.lhs)
init(C::TaoCompose{D,T,R,L,L1,L2,Parametric,<:NoAcceptParams,Parametric,F1,F2}) where {D,T,R,L,L1,L2,F1,F2} = init(C.rhs)

function adjoint(C::TaoCompose{D,T,R,Linear,Linear,Linear,P,P1,P2,F1,F2,O}) where
    {
        D,T,R,P,P1,P2,F1,F2,O
    }
    lhs_out = adjoint(C.rhs)
    rhs_out = adjoint(C.lhs)
    T1 = typeof(lhs_out)
    T2 = typeof(rhs_out)
    return TaoCompose{R,T,D,Linear,Linear,Linear,P,P2,P1,T1,T2,O}(lhs_out, rhs_out)
end

function (C::TaoCompose{D,T,R,L,L1,L2,Parametric,Parametric,Parametric,F1,F2,LeftFirst})(θ::V) where
    {
        D,T,R,L,L1,L2,F1,F2,V<:AbstractVector
    }
    lhs_out = C.lhs(@view θ[1:nparams(C.lhs)])
    rhs_out = C.rhs(@view θ[nparams(C.lhs)+1:nparams(C.lhs)+nparams(C.rhs)])
    T1 = typeof(lhs_out)
    T2 = typeof(rhs_out)
    return TaoCompose{D,T,R,L,L1,L2,Parameterized,Parameterized,Parameterized,T1,T2,LeftFirst}(lhs_out, rhs_out)
end

function (C::TaoCompose{D,T,R,L,L1,L2,Parametric,Parametric,Parametric,F1,F2,RightFirst})(θ::V) where
    {
        D,T,R,L,L1,L2,F1,F2,V<:AbstractVector
    }
    lhs_out = C.lhs(@view θ[nparams(C.lhs)+1:nparams(C.lhs)+nparams(C.rhs)])
    rhs_out = C.rhs(@view θ[1:nparams(C.lhs)])
    T1 = typeof(lhs_out)
    T2 = typeof(rhs_out)
    return TaoCompose{D,T,R,L,L1,L2,Parameterized,Parameterized,Parameterized,T1,T2,LeftFirst}(lhs_out, rhs_out)
end

function (C::TaoCompose{D,T,R,L,L1,L2,Parametric,Parametric,P,F1,F2,O})(θ::V) where
    {
        D,T,R,L,L1,L2,P<:NoAcceptParams,F1,F2,V<:AbstractVector,O<:ParamOrder
    }
    lhs_out = C.lhs(θ)
    T1 = typeof(lhs_out)
    return TaoCompose{D,T,R,L,L1,L2,Parameterized,Parameterized,P,T1,F2,O}(lhs_out, C.rhs)
end

function (C::TaoCompose{D,T,R,L,L1,L2,Parametric,P,Parametric,F1,F2,O})(θ::V) where
    {
        D,T,R,L,L1,L2,P<:NoAcceptParams,F1,F2,V<:AbstractVector,O<:ParamOrder
    }
    rhs_out = C.rhs(θ)
    T2 = typeof(rhs_out)
    return TaoCompose{D,T,R,L,L1,L2,Parameterized,P,Parameterized,F1,T2,O}(C.lhs, rhs_out)
end

function (C::TaoCompose{D,T,R,L,L1,L2,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,F2,O})(x::X) where
    {
        D,T,R,L,L1,L2,F1,F2,O,X<:AbstractVecOrMat{D}
    }
    return C.lhs(C.rhs(x))
end