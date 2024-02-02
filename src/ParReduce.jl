export ParReduce

"""
Reduction Operator. Reduce across the given communicator
"""
struct ParReduce{T} <: ParOperator{T,T,Linear,NonParametric,External}
    comm::MPI.Comm

    ParReduce() = new{Float64}(MPI.COMM_WORLD)
    ParReduce(T::DataType) = new{T}(MPI.COMM_WORLD)
    ParReduce(T::DataType, comm::MPI.Comm) = new{T}(comm)
    ParReduce(comm::MPI.Comm) = new{Float64}(comm)
end

(A::ParReduce{T})(x::X) where {T,X<:AbstractVector{T}} = MPI.Allreduce(x, MPI.SUM, A.comm)
(A::ParReduce{T})(x::X) where {T,X<:AbstractArray{T}} = MPI.Allreduce(x, MPI.SUM, A.comm)

function ChainRulesCore.rrule(A::ParReduce{T}, x::X) where {T,X<:AbstractVector{T}}
    op_out = A(x)
    function pullback(op)
        return NoTangent(), op # TODO: Fix this
    end
    return op_out, pullback
end

function ChainRulesCore.rrule(A::ParReduce{T}, x::X) where {T,X<:AbstractArray{T}}
    op_out = A(x)
    function pullback(op)
        device = get_device(x)
        if device == "cpu"
            return NoTangent(), A(op)
        elseif device == "gpu"
            return NoTangent(), A(op |> gpu) |> cpu
        end
    end
    return op_out, pullback
end
