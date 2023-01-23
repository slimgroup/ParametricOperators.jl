export ParBroadcasted

struct ParBroadcasted{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    op::F
    comm::MPI.Comm
    is_root::Bool
    ParBroadcasted(op, comm) = new{DDT(op),RDT(op),linearity(op),parametricity(op),typeof(op)}(op, comm, MPI.Comm_rank(comm) == 0)
end

Domain(A::ParBroadcasted) = Domain(A.op)
Range(A::ParBroadcasted) = Range(A.op)
adjoint(A::ParBroadcasted) = ParBroadcasted(adjoint(A.op), A.comm)
nparams(A::ParBroadcasted) = A.is_root ? nparams(A.op) : 0
init(A::ParBroadcasted) = A.is_root ? init(A.op) : []
params(A::ParBroadcasted) = A.is_root ? params(A.op) : []

children(A::ParBroadcasted) = [A.op]
from_children(A::ParBroadcasted, cs) = ParBroadcasted(cs[1], A.comm)

function (A::ParBroadcasted{D,R,L,Parametric,F})(params) where {D,R,L,F}
    params = MPI.bcast(params, 0, A.comm)
    return ParBroadcasted(A.op(params), A.comm)
end

(A::ParBroadcasted{D,R,L,<:Applicable,F})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.op(x)
(A::ParBroadcasted{D,R,L,<:Applicable,F})(x::X) where {D,R,L,F,X<:AbstractMatrix{D}} = A.op(x)
*(x::X, A::ParBroadcasted{D,R,Linear,<:Applicable,F}) where {D,R,F,X<:AbstractMatrix{D}} = x*A.op
