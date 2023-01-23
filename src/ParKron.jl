export ParSeparableOperator, ParKron, ⊗

"""
Abstract type for operators that are separable (i.e. representable as Kronecker products)
"""
abstract type ParSeparableOperator{D,R,P,T} <: ParLinearOperator{D,R,P,T} end

"""
Get the order of operations of a separable operator
"""
order(::ParSeparableOperator) = throw(ParException("Unimplemented"))

"""
Kronecker product of linear operators
"""
struct ParKron{D,R,P,F,N} <: ParSeparableOperator{D,R,P,Internal}
    ops::F # Operators in Kronecker order (e.g. B ⊗ A ⟹   ops = [B, A])
    order::Vector{Int} # Order of operator application
    perms::Vector{NTuple{N, Int}} # List of permutations between input/output and intermediates
    shapes_in::Vector{NTuple{N, Int}} # List of permuted input shapes
    shapes_out::Vector{NTuple{N, Int}} # List of permuted output shapes
    
    function ParKron(ops::ParLinearOperator...)
        
        ops = collect(ops)
        N = length(ops)

        order = nothing
        perms = nothing
        shapes_in = nothing
        shapes_out = nothing

        @ignore_derivatives begin 

            # Compute operator application order
            order = zeros(Int, N)
            
            # Find the "innermost" type
            T = reduce(subset_type, map(DDT, ops))
            for i in 1:N
                
                # Find indices of operators not currently in the order with DDT = T
                candidates = filter(j -> DDT(ops[j]) == T && j ∉ order, 1:N)

                # Find the "innermost" range type and filter candidates
                R = reduce(subset_type, map(j -> RDT(ops[j]), candidates))
                filter!(j -> RDT(ops[j]) == R, candidates)

                # Find the smallest range of the remaining candidates and filter
                min_range = minimum(map(j -> Range(ops[j]), candidates))
                filter!(j -> Range(ops[j]) == min_range, candidates)

                # Select the operator in slot closest to contiguous dimension
                @assert length(candidates) > 0 "Invalid datatype combination in ParKron"
                order[i] = candidates[end]

                # Update domain type
                T = R

            end

            # From order of operators, compute input/output shapes
            shapes_in = Vector{NTuple{N, Int}}(undef, N)
            shapes_out = Vector{NTuple{N, Int}}(undef, N)
            shapes_in[1] = ntuple(i -> Domain(ops[N-i+1]), N)
            shapes_out[N] = ntuple(i -> Range(ops[N-i+1]), N)

            for i in 1:N-1
                o = order[i]
                d = N-o+1
                shape = collect(shapes_in[i])
                shape[d] = Range(ops[o])
                shapes_in[i+1] = tuple(shape...)
                shapes_out[i] = tuple(shape...)
            end

            # Update shapes from order
            for i in 1:N
                if i > 1
                    shapes_in[i] = tuple(circshift(collect(shapes_in[i]), order[i])...) 
                end
                if i < N
                    shapes_out[i] = tuple(circshift(collect(shapes_out[i]), order[i])...) 
                end
            end

            # Compute permuations from operator order
            perms = Vector{NTuple{N, Int}}(undef, N+1)
            for i in 1:N+1
                if i == 1
                    perms[i] = tuple(circshift(1:N, order[1])...)
                elseif i == N+1
                    perms[i] = tuple(circshift(1:N, -order[N])...)
                else
                    perms[i] = tuple(circshift(1:N, order[i]-order[i-1])...)
                end
            end
        end

        D = DDT(ops[order[1]])
        R = RDT(ops[order[N]])
        P = reduce(promote_parametricity, map(parametricity, ops))
        return new{D,R,P,typeof(ops),N}(ops, order, perms, shapes_in, shapes_out)
    end

    function ParKron(ops::Vector{<:ParLinearOperator}, order, perms, shapes_in, shapes_out)
        N = length(ops)
        D = DDT(ops[order[1]])
        R = RDT(ops[order[N]])
        P = reduce(promote_parametricity, map(parametricity, ops))
        return new{D,R,P,typeof(ops),N}(ops, order, perms, shapes_in, shapes_out)
    end
end

⊗(A::ParLinearOperator, B::ParLinearOperator) = ParKron(A, B)
⊗(A::ParKron, B::ParLinearOperator) = ParKron(A.ops..., B)
⊗(A::ParLinearOperator, B::ParKron) = ParKron(A, B.ops...)
⊗(A::ParKron, B::ParKron) = ParKron(A.ops..., B.ops...)

Domain(A::ParSeparableOperator) = prod(map(Domain, children(A)))
Range(A::ParSeparableOperator) = prod(map(Range, children(A)))
children(A::ParKron) = A.ops
from_children(::ParKron, cs) = ParKron(cs...)

function adjoint(A::ParKron)
    ops = collect(map(adjoint, A.ops))
    N = length(ops)
    order = reverse(A.order)
    perms = Vector{NTuple{N, Int}}(undef, N+1)
    @ignore_derivatives begin
        for i in 1:N+1
            if i == 1
                perms[i] = tuple(circshift(1:N, order[1])...)
            elseif i == N+1
                perms[i] = tuple(circshift(1:N, -order[N])...)
            else
                perms[i] = tuple(circshift(1:N, order[i]-order[i-1])...)
            end
        end
    end
    shapes_in = reverse(A.shapes_out)
    shapes_out = reverse(A.shapes_in)
    return ParKron(ops, order, perms, shapes_in, shapes_out)
end

function (A::ParKron{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractMatrix{D}}

    batch_size = size(x)[2]

    # Apply each operator in order by permuting x to be contiguous along that
    # operator's dimension of application
    x = reshape(x, A.shapes_in[1]..., batch_size)
    for i in 1:N
        x = permutedims(x, tuple(A.perms[i]..., N+1))
        x = reshape(x, size(x)[1], prod(size(x)[2:N+1]))
        x = A.ops[A.order[i]]*x
        x = reshape(x, A.shapes_out[i]..., batch_size)
    end
    return reshape(x, prod(size(x)[1:N]), batch_size)
end

function (A::ParKron{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractVector{D}}
    x = reshape(x, length(x), 1)
    return vec(A(x))
end

function distribute(A::ParKron, dims...; parent_comm=MPI.COMM_WORLD)
    
    # Transform kronecker of operators to product of operators kroneckered with
    # identity operator and repartitions. E.g.
    #
    # K = C ⊗ B ⊗ A
    #
    # becomes
    #
    # K_dist = R_{C->out}*(C ⊗ I_B ⊗ I_A)*R_{B->A}*(I_C ⊗ B ⊗ I_A)*R_{A->B}*(I_C ⊗ I_B ⊗ A)*R_{in->A}
    #
    # assuming the application order is C -> B -> A.
    
    ops = A.ops
    order = A.order
    N = length(ops)

    # Get communicators from application order
    dims = collect(dims)
    comms = Vector{MPI.Comm}(undef, N+1)
    comms[1] = MPI.Cart_create(parent_comm, Int32.(dims))

    for i in 1:N
        dims_i = copy(dims)
        o = order[i]
        d = N-o+1
        dims_i[d] = 1
        dims_i[mod1(d+1,N)] *= dims[d]
        comms[i+1] = MPI.Cart_create(parent_comm, dims_i)
    end

    # From shapes and communicators, create repartition operators
    Rs = Vector{ParRepartition}(undef, N+1)
    Rs[1]   = ParRepartition(DDT(A), comms[1], comms[2], A.shapes_in[1])
    Rs[N+1] = ParRepartition(RDT(A), comms[N+1], comms[1], A.shapes_out[N])

    for i in 2:N
        comm_in = comms[i]
        comm_out = comms[i+1]
        global_size = A.shapes_in[i]
        T = DDT(ops[order[i]])
        Rs[i] = ParRepartition(T, comm_in, comm_out, global_size)
    end

    # From shapes and communicators, create single dimension operators
    As = Vector{ParKron}(undef, N)
    shape = collect(A.shapes_in[1])
    for i in 1:N
        o = order[i]
        d = N-o+1
        comm_i = comms[i+1]
        dims_i, _, coords_i = MPI.Cart_get(comm_i)
        Ai = ParBroadcasted(ops[o], comm_i)
        
        Is_dim_lower = []
        Is_dim_upper = []
        if d > 1
            Is_dim_lower = [ParDistributed(ParIdentity(DDT(ops[o]), shape[j]), (1, dims_i[j]), (0, coords_i[j])) for j in 1:d-1]
        end
        if d < N
            Is_dim_upper = [ParDistributed(ParIdentity(DDT(ops[o]), shape[j]), (1, dims_i[j]), (0, coords_i[j])) for j in d+1:N]
        end

        As[i] = ParKron(reverse(Is_dim_upper)..., Ai, reverse(Is_dim_lower)...)
        shape[d] = Range(ops[o])
    end

    ops_out = Vector{ParLinearOperator}()
    pushfirst!(ops_out, Rs[1])
    for i in 1:N
        pushfirst!(ops_out, As[i])
        pushfirst!(ops_out, Rs[i+1])
    end

    return ∘(ops_out...)

end
