export merkle_hash, transforms

"""
Computes the Merkle hash of a given subtree for use with AST optimization.
"""
function merkle_hash(A::ParOperator{D,R,L,P,External}) where {D,R,L,P}
    # Name of A without types
    op_str = "$(Base.typename(typeof(A)).wrapper)"
    
    # Domain and range types
    op_str *= "_DDT=$(D)_RDT=$(R)"

    # Domain and range values
    op_str *= "_Domain=$(Domain(A))_Range=$(Range(A))"

    return hash(op_str)
end

function merkle_hash(A::ParOperator{D,R,L,P,Internal}) where {D,R,L,P}
    # Combine hashes of children
    hash_str = foldl(*, map(c -> "$(merkle_hash(c))", children(A)))
    return hash(hash_str)
end

transforms(A::ParOperator) = [A]