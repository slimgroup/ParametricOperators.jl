using MPI

uid() = uuid4(GLOBAL_RNG)

subtype(t::T, u::T) where {D<:Number,T<:Type{D}} = t
subtype(t::T, u::U) where {R<:Real,T<:Type{R},U<:Type{Complex{R}}} = t
subtype(u::U, t::T) where {R<:Real,T<:Type{R},U<:Type{Complex{R}}} = u

function print0(x::Any, comm::MPI.Comm)
    if MPI.Comm_rank(comm) == 0
        print("$(x)")
    end
end

println0(x::Any, comm::MPI.Comm) = print0("$(x)\n", comm)

function printstyled0(x::Any, comm::MPI.Comm; kwargs...)
    if MPI.Comm_rank(comm) == 0
        printstyled("$(x)", kwargs...)
    end
end