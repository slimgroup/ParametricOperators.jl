export ParSeparableOperator, ParKron, ⊗

abstract type ParSeparableOperator{D,R,P,T} <: ParLinearOperator{D,R,P,T} end

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