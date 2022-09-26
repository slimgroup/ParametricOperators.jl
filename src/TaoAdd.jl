export TaoAdd

struct TaoAdd{
    D<:Number,
    R<:Number,
    L<:Linearity,
    L1<:Linearity,
    L2<:Linearity,
    P<:Parametricity,
    P1<:Parametricity,
    P2<:Parametricity,
    F1<:TaoOperator{D,R,L1,P1,<:NodeType},
    F2<:TaoOperator{D,R,L2,P2,<:NodeType}
} <: TaoOperator{D,R,L,P,Internal}
    lhs::F1
    rhs::F2
end

function +(lhs::F1, rhs::F2) where {D,R,L1,L2,P1,P2,F1<:TaoOperator{D,R,L1,P1,<:NodeType},F2<:TaoOperator{D,R,L2,P2,<:NodeType}}
    @assert dims_compatible(Domain(lhs), Domain(rhs))
    @assert dims_compatible(Range(lhs), Range(rhs))
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
    return TaoAdd{D,R,L,L1,L2,P,P1,P2,F1,F2}(lhs, rhs)
end

Domain(A::TaoAdd) = Domain(A.lhs)
Range(A::TaoAdd) = Range(A.lhs)
children(A::TaoAdd) = [A.lhs, A.rhs]
id(A::TaoAdd) = "[$(id(A.lhs))]_add_[$(id(A.rhs))]"

init(A::TaoAdd{D,R,L,L1,L2,Parametric,Parametric,Parametric}) where {D,R,L,L1,L2} = return mapreduce(init, vcat, children(A))

init(A::TaoAdd{D,R,L,L1,L2,Parametric,Parametric,<:NoAcceptParams}) where {D,R,L,L1,L2} = init(A.lhs)

init(A::TaoAdd{D,R,L,L1,L2,Parametric,<:NoAcceptParams,Parametric}) where {D,R,L,L1,L2} = init(A.rhs)

adjoint(A::TaoAdd) = adjoint(A.lhs) + adjoint(A.rhs)

function (A::TaoAdd{D,R,L,L1,L2,Parametric,Parametric,Parametric,F1,F2})(θ::V) where 
    {D,R,L,L1,L2,F1,F2,V<:AbstractVecOrMat}
    lhs_out = A.lhs(@view θ[1:nparams(A.lhs)])
    rhs_out = A.rhs(@view θ[nparams(A.lhs)+1:nparams(A.lhs)+nparams(A.rhs)])
    return TaoAdd{D,R,L,L1,L2,Parameterized,Parameterized,Parameterized,typeof(lhs_out),typeof(rhs_out)}(lhs_out, rhs_out)
end

function (A::TaoAdd{D,R,L,L1,L2,Parametric,Parametric,P,F1,F2})(θ::V) where 
    {D,R,L,L1,L2,P<:NoAcceptParams,F1,F2,V<:AbstractVecOrMat}
    lhs_out = A.lhs(θ)
    return TaoAdd{D,R,L,L1,L2,Parameterized,Parameterized,P,typeof(lhs_out),F2}(lhs_out, A.rhs)
end

function (A::TaoAdd{D,R,L,L1,L2,Parametric,P,Parametric,F1,F2})(θ::V) where 
    {D,R,L,L1,L2,P<:NoAcceptParams,F1,F2,V<:AbstractVecOrMat}
    rhs_out = A.rhs(θ)
    return TaoAdd{D,R,L,L1,L2,Parameterized,P,Parameterized,F1,typeof(rhs_out)}(A.lhs, rhs_out)
end

(A::TaoAdd{D,R,L,L1,L2,<:NoAcceptParams,<:NoAcceptParams,<:NoAcceptParams,F1,F2})(x::X) where
    {D,R,L,L1,L2,F1,F2,X<:AbstractVecOrMat{D}} = A.lhs(x) + A.rhs(x)