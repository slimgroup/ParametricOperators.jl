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
    function ParKron(ops...)

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
                differences = map(j -> Range(ops[j]) - Domain(ops[j]), candidates)
                min_difference = minimum(differences)
                candidates = filter(j -> Range(ops[j]) - Domain(ops[j]) == min_difference, candidates)

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

    function ParKron(D,R,P,ops,order)
        return new{D,R,P,typeof(ops),length(ops)}(ops, order)
    end
end

kron(A::ParLinearOperator, B::ParLinearOperator) = ParKron(A, B)
kron(A::ParKron, B::ParLinearOperator) = ParKron(A.ops..., B)
kron(A::ParLinearOperator, B::ParKron) = ParKron(A, B.ops...)
kron(A::ParKron, B::ParKron) = ParKron(A.ops..., B.ops...)
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

"""
Multiplication of two separable operators gives the ability to apply the mixed-product
property.
"""
function transforms(A::ParCompose{D,R,Linear,P,<:AbstractVector{<:ParSeparableOperator},2}) where {D,R,P}
    
    A_lhs = A.ops[1]
    A_rhs = A.ops[2]
    cs_lhs = children(A_lhs)
    cs_rhs = children(A_rhs)
    N = length(cs_lhs)
    
    # If there are the same number of operators and they have matching dimensions,
    # it is valid to apply the rule
    if length(cs_rhs) == N && all(DDT(l) == RDT(r) && Domain(l) == Range(r) for (l, r) in zip(cs_lhs, cs_rhs))
        return Channel() do channel
            for select in Iterators.product([[true, false] for _ in 1:N]...)
                
                # If no operators are selected from rhs, keep going
                if !any(select)
                    continue
                end
                
                # Move operators from rhs -> lhs, replacing rhs w/ identity
                ops_out_lhs = [select[i] ? cs_lhs[i]*cs_rhs[i] : cs_lhs[i] for i in 1:N]
                ops_out_rhs = [select[i] ? ParIdentity(DDT(cs_rhs[i]), Domain(cs_rhs[i])) : cs_rhs[i] for i in 1:N]
                
                # If we moved all operators, only return lhs, otherwise return
                # combination of both
                if all(select)
                    for op in transforms(rebuild(A_lhs, ops_out_lhs))
                        put!(channel, op)
                    end
                else
                    lhs_out = rebuild(A_lhs, ops_out_lhs)
                    rhs_out = rebuild(A_rhs, ops_out_rhs)
                    for ops in Iterators.product(transforms(lhs_out), transforms(rhs_out))
                        put!(channel, ops[1]*ops[2])
                    end
                end
            end
        end
    else
        return []
    end
end

function (A::ParKron{D,R,<:Applicable,F,N})(x::X) where {D,R,F,N,X<:AbstractMatrix{D}}
    
    # Reshape to input shape
    b = size(x)[2]
    s = reverse(collect(map(Domain, A.ops)))
    x = reshape(x, s..., b)

    # Apply operators in order, permuting to enforce leading dim of x to
    # align with current operator
    x = rotate_dims_batched(x, -(N-A.order[1]))

    for i in 1:N
        o = A.order[i]
        s = size(x)
        x = as_matrix(x)
        Ai = A.ops[o]
        x = Ai*x
        x = reshape(x, Range(Ai), s[2:end]...)
        if i < N
            x = rotate_dims_batched(x, N-o-(N-A.order[i+1]))
        else
            x = rotate_dims_batched(x, o)
        end
    end

    nelem = prod(size(x))
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

"""
Distributes Kronecker product over the given communicator
"""
function distribute(A::ParKron, dims_in, dims_out=dims_in, parent_comm=MPI.COMM_WORLD)

    comm_in  = MPI.Cart_create(parent_comm, dims_in)
    comm_out = MPI.Cart_create(parent_comm, dims_out)

    dims, _, _ = MPI.Cart_get(comm_in)
    N = length(dims)
    @assert length(A.ops) == N

    size_curr = collect(map(Domain, reverse(A.ops)))
    comm_prev = comm_in

    ops = []

    for i in 1:N

        @show size_curr

        # Get operator i
        o = A.order[i]
        d = N-o+1
        Ai = A.ops[o]

        # Compute size of dims for communicator
        dims_i = copy(dims)
        dims_i[d] = 1
        dims_i[mod1(d+1, N)] *= dims[d]
        comm_i = MPI.Cart_create(parent_comm, dims_i)
        coords_i = MPI.Cart_coords(comm_i)

        # Create repartition operator
        pushfirst!(ops, ParRepartition(DDT(Ai), comm_prev, comm_i, tuple(size_curr...)))

        # Create Kronecker w/ distributed identities
        idents_dim_lower = []
        idents_dim_upper = []
        
        for j in d+1:N
            pushfirst!(idents_dim_lower, ParDistributed(ParIdentity(DDT(Ai), size_curr[j]), coords_i[j], dims_i[j]))
        end
        for j in 1:d-1
            pushfirst!(idents_dim_upper, ParDistributed(ParIdentity(DDT(Ai), size_curr[j]), coords_i[j], dims_i[j]))
        end

        pushfirst!(ops, ParKron(idents_dim_upper..., ParBroadcasted(Ai, comm_i), idents_dim_lower...))

        size_curr[d] = Range(Ai)
        comm_prev = comm_i
    end

    pushfirst!(ops, ParRepartition(RDT(A.ops[A.order[end]]), comm_prev, comm_out, tuple(size_curr...)))

    return ParCompose(ops...)
end