export ParSeparableOperator, ParKron, ⊗
export reorder

abstract type ParSeparableOperator{D,R,P,T} <: ParLinearOperator{D,R,P,T} end

"""
Get the order of operator application in a separable operator.
"""
order(::ParSeparableOperator) = throw(ParException("Unimplemented"))

"""
Kronecker product operator.
"""
struct ParKron{D,R,P,F,N} <: ParSeparableOperator{D,R,P,Internal}
    ops::F
    order::Vector{Int}

    function ParKron(ops::Vector,order::Vector)

        D = DDT(ops[order[1]])
        R = RDT(ops[order[end]])
        P = foldl(promote_parametricity, map(parametricity, ops))

        return new{D,R,P,typeof(ops),length(ops)}(ops, order)
    end

    function ParKron(ops...)
        # Brute force application from right to left because sometimes reordering 
        # causing more repartitions which are much more expensive as of now for desired applications
        # rn in the fno code the subtype computed here is always applied first, so no repartition.
        # for example: we need to take RFFT before the identity in fno

        # Collect operators into a vector
        ops = collect(ops)
        N = length(ops)

        # We can't Kronecker only a single operator
        if N == 1
            return ops[1]
        end

        # Find the domain type which is the most "sub" type
        DDTs = map(DDT, ops)
        RDTs = map(RDT, ops)
        T = foldl(subset_type, DDTs)

        # Compute operator application order
        order = zeros(Int, N)
        @ignore_derivatives begin
            for i in 1:N

                # Find all operator indices with the current domain type
                candidates = filter(j -> DDTs[j] == T && j ∉ order, 1:N)

                # From the candidates, find the range type which is the most
                # "sub" type and filter candidates
                R = mapreduce(j -> RDTs[j], subset_type, candidates)
                candidates = filter(j -> RDTs[j] == R, candidates)

                # Filter the candidates to those which minimize the array size
                # on output
                differences = map(j -> Domain(ops[j]) - Range(ops[j]), candidates)
                max_difference = maximum(differences)
                candidates = filter(j -> (Domain(ops[j]) - Range(ops[j])) == max_difference, candidates)

                # Finally, if there is more than one potential candidate, select
                # in right-to-left order
                order[i] = candidates[end]
                T = RDTs[order[i]]
            end
        end

        D = DDT(ops[order[1]])
        R = RDT(ops[order[N]])
        P = foldl(promote_parametricity, map(parametricity, ops))

        return new{D,R,P,typeof(ops),N}(ops, order)
    end

    function ParKron(D::DataType,R::DataType,P,ops,order)
        return new{D,R,P,typeof(ops),length(ops)}(ops, order)
    end
end

kron(A::ParLinearOperator, B::ParLinearOperator) = ParKron(A, B)
kron(A::ParKron, B::ParLinearOperator) = ParKron(A.ops..., B)
kron(A::ParLinearOperator, B::ParKron) = ParKron(A, B.ops...)
⊗(A::ParKron, B::ParKron) = ParKron(A.ops..., B.ops...)
⊗(A::ParLinearOperator, B::ParLinearOperator) = kron(A, B)

Domain(A::ParSeparableOperator) = prod(map(Domain, children(A)))
Range(A::ParSeparableOperator) = prod(map(Range, children(A)))

children(A::ParKron) = A.ops
rebuild(::ParKron, cs) = ParKron(cs...)
adjoint(A::ParKron{D,R,P,F,N}) where {D,R,P,F,N} = ParKron(R,D,P,collect(map(adjoint, A.ops)), reverse(A.order))
order(A::ParKron) = A.order

"""
Change application order of Kronecker product. Throws an error if the
given order would result in an invalid type sequence. TODO: add the
option to force type conversion?
"""
function reorder(A::ParKron{D,R,P,F,N}, ord) where {D,R,P,F,N}
    for i in 1:N-1
        if RDT(A.ops[ord[i]]) != DDT(A.ops[ord[i]])
            throw(ParException("Invalid order $ord for Kronecker product $A. Types do not agree"))
        end
    end
    return ParKron(D,R,P,A.ops,ord)
end

"""
Complexity of a separable operator is given by taking the complexity
of applying an individual operator an multiplying it by the size of
the rest of the tensor (# of cols in matrix form).
"""
function complexity(A::ParSeparableOperator)
    c = 0.0
    p = prod(map(Domain, children(A)))
    cs = children(A)
    for o in order(A)
        # (Complexity of mvp) * (# cols in matrix)
        c += complexity(cs[o])*(p÷Domain(cs[o]))
        p = (p ÷ Domain(cs[o])) * Range(cs[o])
    end
    return c
end

"""
The hash of a separable operator also depends on application order.
"""
function merkle_hash(A::ParSeparableOperator)
    # Combine hashes of children
    hash_str = foldl(*, map(c -> "$(merkle_hash(c))", children(A)))

    # Add hash of order (cast to Int64 to prevent differing types causing differing hashes)
    hash_str *= "$(hash(Int64.(order(A))))"

    return hash(hash_str)
end

"""
Transformations for Kronecker product:
    - Inserting parentheses
    - Reordering operators
"""
function transforms(A::ParKron)
    n = length(A.ops)

    return Channel() do channel

        # Insert parens
        if n > 2
            for paren_length in 2:n-1
                for i in 1:n-paren_length+1

                    left_ops = [A.ops[j] for j in 1:i-1]
                    middle_ops = [A.ops[j] for j in i:i+paren_length-1]
                    right_ops = [A.ops[j] for j in i+paren_length:n]

                    # TODO: This does not cover the case of no parents on some op groups...
                    @match (length(left_ops), length(middle_ops), length(right_ops)) begin
                        (0, n, m) => begin
                            M = ParKron(middle_ops...)
                            R = ParKron(right_ops...)
                            for (M_t, R_t) in Iterators.product(transforms(M), transforms(R))
                                put!(channel, ParKron(M_t, R_t))
                            end
                        end
                        (n, m, 0) => begin
                            L = ParKron(left_ops...)
                            M = ParKron(middle_ops...)
                            for (L_t, M_t) in Iterators.product(transforms(L), transforms(M))
                                put!(channel, ParKron(L_t, M_t))
                            end
                        end
                        (n, m, k) => begin
                            L = ParKron(left_ops...)
                            M = ParKron(middle_ops...)
                            R = ParKron(right_ops...)
                            for (L_t, M_t, R_t) in Iterators.product(transforms(L), transforms(M), transforms(R))
                                put!(channel, ParKron(L_t, M_t, R_t))
                            end
                        end
                    end
                end
            end
        end

        # Permute order
        for ord in permutations(collect(1:n))

            # Try-catch here captures the exception thrown by reorder() if the output is invalid
            try
                A_reorder = reorder(A, ord)
                put!(channel, A_reorder)
            catch
            end
        end
    end
end

function (A::ParKron{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractMatrix{D}}
    # comm = MPI.COMM_WORLD
    # rank = MPI.Comm_rank(comm)
    # rank == 0 && println(x)
    # rank == 0 && println("Here")
    # Reshape to inpu t shape
    # rank == 0 && println("0:", typeof(x))
    b = size(x)[2]
    s = reverse(collect(map(Domain, A.ops)))
    # rank == 0 && println("Size: ", s)
    x = reshape(x, s..., b)
    # rank == 0 && println("0.5:", typeof(x))

    # Apply operators in order, permuting to enforce leading dim of x to
    # align with current operator
    # rank == 0 && println("Pre Rotation: ", x)
    x = rotate_dims_batched(x, -(N-A.order[1]))
    # rank == 0 && println("Post rotation: ", x)
    # rank == 0 && println("1:", typeof(x))
    # rank == 0 && println("Starting Loop")
    for i in 1:N
        o = A.order[i]
        s = size(x)
        x = as_matrix(x)
        # rank == 0 && println("2:", typeof(x))
        Ai = A.ops[o]
        # rank == 0 && println(i, " ", x, " ", typeof(Ai), " ", Range(Ai), "x", Domain(Ai))
        # rank == 0 && println("3:", typeof(x), typeof(Ai))
        x = Ai*x
        # rank == 0 && println(i, " Post operation: ", x)
        # rank == 0 && println("4:", typeof(x))
        x = reshape(x, Range(Ai), s[2:end]...)
        # rank == 0 && println("5:", typeof(x))
        if i < N
            # rank == 0 && println("rotation around: ", N-o-(N-A.order[i+1]))
            x = rotate_dims_batched(x, N-o-(N-A.order[i+1]))
        else
            # rank == 0 && println("rotation around: ", -o)
            x = rotate_dims_batched(x, -o)
        end
        # rank == 0 && println(i, " ", reshape(x, 4, 1))
        # rank == 0 && println("6:", typeof(x))
    end

    nelem = prod(size(x))
    # rank == 0 && println(reshape(x, nelem÷b, b))
    return reshape(x, nelem÷b, b)
end

(A::ParKron{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractVector{D}} =
    vec(A(reshape(x, length(x), 1)))

function latex_string(A::ParKron{D,R,P,F,N}) where {D,R,P,F,N}
    child_eqns = [latex_string(c) for c in children(A)]
    if ast_location(A.ops[1]) == Internal
        out = "($(child_eqns[1]))"
    else
        out = child_eqns[1]
    end
    for i in 2:N
        if ast_location(A.ops[i]) == Internal
            out *= "\\otimes($(child_eqns[i]))"
        else
            out *= "\\otimes $(child_eqns[i])"
        end
    end
    return out
end

rebuild(A::ParBroadcasted{D,R,L,Parametric,F}, cs) where {D,R,L,F<:ParKron} = rebuild(A.op, collect(map(c -> parametricity(c) == Parametric ? ParBroadcasted(c, A.comm, A.root) : c, children(cs[1]))))

"""
Distributes Kronecker product over the given dimensions
"""
function distribute(A::ParKron, dims_in::Vector{Int64}, dims_out::Vector{Int64}=dims_in, parent_comm=MPI.COMM_WORLD)
    comm_in  = MPI.Cart_create(parent_comm, dims_in)
    comm_out = MPI.Cart_create(parent_comm, dims_out)

    return distribute(A, comm_in, comm_out, parent_comm)
end

"""
Distributes Kronecker product over the given communicator
"""
function distribute(A::ParKron, comm_in::MPI.Comm, comm_out::MPI.Comm, parent_comm=MPI.COMM_WORLD)

    dims, _, _ = MPI.Cart_get(comm_in)
    dims_out, _, _ = MPI.Cart_get(comm_out)

    N = length(dims)
    @assert length(A.ops) == N

    size_curr = collect(map(Domain, reverse(A.ops))) # [30,20] 
    comm_prev = comm_in
    dims_prev = dims

    ops = []

    # if MPI.Comm_rank(comm_in) == 0
        # println(size_curr, A.order)
    # end

    for i in 1:N

        # Get operator i
        o = A.order[i] # o is the index where the ith operate to execute is stored at
        d = N-o+1 # this is just i, not sure why its here ?? (maybe bc parKron rearranger A.order, so we need this)
        Ai = A.ops[o] # The i th operator

        # Compute size of dims for communicator
        dims_i = copy(dims)
        dims_i[d] = 1
        dims_i[mod1(d+1, N)] *= dims[d]
        # if MPI.Comm_rank(comm_in) == 0 # Move the distribution for i th operator to the i+1 th operator
            # println(d)
            # println(dims_i)
        # end
        comm_i = MPI.Cart_create(parent_comm, dims_i)
        rank = MPI.Comm_rank(comm_i)
        coords_i = MPI.Cart_coords(comm_i)
        # if MPI.Comm_rank(comm_in) == 1 # Move the distribution for i th operator to the i+1 th operator
            # println(coords_i)
        # end

        # MPI.Comm_rank(comm_in) == 0 && println("Iteration: ", i, ". d: ", d, " ", dims_i, " ", typeof(Ai))

        # # Skip this iteration if it does nothing
        # (typeof(Ai) <: ParIdentity) && (MPI.Comm_rank(comm_in) == 0) && println("Skipping loop at Iter: ", i)
        (typeof(Ai) <: ParIdentity) && continue

        # Create repartition operator if data is distributed differently than expected
        !isequal(dims_prev, dims_i) && pushfirst!(ops, ParRepartition(DDT(Ai), comm_prev, comm_i, tuple(size_curr...)))
        !isequal(dims_prev, dims_i) && (MPI.Comm_rank(comm_in) == 0) && println("Pushing a repartition inside @ iteration ", i, " ", dims_prev, dims_i)
        # if MPI.Comm_rank(comm_in) == 0
            # println(length(ops))
        # end
        # Create Kronecker w/ distributed identities
        idents_dim_lower = []
        idents_dim_upper = []

        for j in d+1:N
            # println(size_curr[j], " Lower @ Rank ", MPI.Comm_rank(comm_i))
            pushfirst!(idents_dim_lower, ParDistributed(ParIdentity(DDT(Ai), size_curr[j]), coords_i[j], dims_i[j]))
            # rank == 0 && println("Lower: ", Range(idents_dim_lower[1]), "x", Domain(idents_dim_lower[1]))
        end
        for j in 1:d-1
            # println(size_curr[j], " Upper @ Rank ", MPI.Comm_rank(comm_i))
            pushfirst!(idents_dim_upper, ParDistributed(ParIdentity(DDT(Ai), size_curr[j]), coords_i[j], dims_i[j]))
            # rank == 0 && println("Upper: ", Range(idents_dim_upper[1]), "x", Domain(idents_dim_upper[1]))
        end

        # rank == 0 && println("Actual: ", Range(Ai), "x", Domain(Ai))

        pushfirst!(ops, ParKron(idents_dim_lower..., rebuild(ParBroadcasted(Ai, comm_i), [Ai]), idents_dim_upper...))

        size_curr[d] = Range(Ai)
        comm_prev = comm_i
        dims_prev = dims_i
    end

    !isequal(dims_prev, dims_out) && pushfirst!(ops, ParRepartition(RDT(A.ops[A.order[end]]), comm_prev, comm_out, tuple(size_curr...)))
    # !isequal(dims_prev, dims_out) && (MPI.Comm_rank(comm_in) == 0) && println("Pushing a repartition outside")

    # for op in ops
    #     if MPI.Comm_rank(comm_prev) == 0
            # println(Range(op), " x ", Domain(op))
    #     end
    # end
    return ParCompose(ops...)
end

to_Dict(A::ParKron) = Dict{String, Any}("type" => "ParKron", "of" => map(to_Dict, A.ops), "order" => A.order)

function from_Dict(::Type{ParKron}, d)
    ops = map(from_Dict, d["of"])
    order = d["order"]

    D = DDT(ops[order[1]])
    R = RDT(ops[order[end]])
    P = foldl(promote_parametricity, map(parametricity, ops))

    ParKron(D,R,P,ops,order)
end
