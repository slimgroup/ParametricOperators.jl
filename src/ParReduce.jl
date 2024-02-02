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

function (A::ParReduce{T})(x::X) where {T,X<:AbstractArray{T}}
    device = get_device(x)
    if device == "cpu"
        return MPI.Allreduce(x, MPI.SUM, A.comm)
    elseif device == "gpu"
        return MPI.Allreduce(x |> cpu, MPI.SUM, A.comm) |> gpu
    end
end

function ChainRulesCore.rrule(A::ParReduce{T}, x::X) where {T,X<:AbstractArray{T}}
    op_out = A(x)
    function pullback(op)
        device = get_device(op)
        if device == "cpu"
            return NoTangent(), A(op)
        elseif device == "gpu"
            return NoTangent(), A(op |> cpu) |> gpu
        end
    end
    return op_out, pullback
end
