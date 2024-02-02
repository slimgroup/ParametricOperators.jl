export ParBroadcasted, bcasted

struct ParBroadcasted{D,R,L,P,F} <: ParOperator{D,R,L,P,Internal}
    op::F
    comm::MPI.Comm
    root::Int
    ParBroadcasted(op, comm::Any = MPI.COMM_WORLD, root::Int = 0) = new{DDT(op),RDT(op),linearity(op),parametricity(op),typeof(op)}(op, comm, root)
end

bcasted(A::ParOperator{D,R,L,P,External}, comm = MPI.COMM_WORLD, root = 0) where {D,R,L,P} =
    ParBroadcasted(A, comm, root)

Domain(A::ParBroadcasted) = Domain(A.op)
Range(A::ParBroadcasted) = Range(A.op)
adjoint(A::ParBroadcasted{D,R,Linear,P,F}) where {D,R,P,F} = ParBroadcasted(adjoint(A.op), A.comm, A.root)

function init!(A::ParBroadcasted, d::Parameters)
    if MPI.Comm_rank(A.comm) == A.root
        init!(A.op, d)
    end
end

children(A::ParBroadcasted) = [A.op]
rebuild(A::ParBroadcasted, cs) = ParBroadcasted(cs[1], A.comm, A.root)

function (A::ParBroadcasted{D,R,L,Parametric,F})(params) where {D,R,L,F}
    @assert ast_location(A.op) == External
    θ = MPI.Comm_rank(A.comm) == A.root ? params[A.op] : nothing
    local_params = Dict(A.op => MPI.bcast(θ, A.root, A.comm))
    return ParBroadcasted(A.op(local_params), A.comm, A.root)
end

(A::ParBroadcasted{D,R,L,<:Applicable,F})(x::X) where {D,R,L,F,X<:AbstractVector{D}} = A.op(x)
(A::ParBroadcasted{D,R,L,<:Applicable,F})(x::X) where {D,R,L,F,X<:AbstractMatrix{D}} = A.op(x)
*(x::X, A::ParBroadcasted{D,R,Linear,<:Applicable,F}) where {D,R,F,X<:AbstractMatrix{D}} = x*A.op
+(x::X, A::ParBroadcasted{D,R,Linear,<:Applicable,F}) where {D,R,F,X<:AbstractMatrix{D}} = x+A.op

function ChainRulesCore.rrule(A::ParBroadcasted{D,R,L,Parametric,F}, params) where {D,R,L,F}
    op_out = A(params)
    function pullback(op)
        # θ_global = MPI.Reduce(op.op.params |> cpu, MPI.SUM, A.root, A.comm)
        θ_global = MPI.Reduce(op.op.params, MPI.SUM, A.root, A.comm)

        if MPI.Comm_rank(A.comm) == A.root
            # return NoTangent(), Dict(A.op=>(θ_global |> gpu))
            return NoTangent(), Dict(A.op=>θ_global)
        else
            return NoTangent(), NoTangent()
        end
    end
    return op_out, pullback
end
