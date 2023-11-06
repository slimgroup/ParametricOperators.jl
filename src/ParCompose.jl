export ParCompose

"""
Composition operator.
"""
struct ParCompose{D,R,L,P,F,N} <: ParOperator{D,R,L,P,Internal}
    ops::F
    function ParCompose(ops...)
        ops = collect(ops)
        ops = collect(ops)
        N = length(ops)
        if N == 1
            return ops[1]
        end
        @ignore_derivatives begin
            for i in 1:N-1
                @assert Domain(ops[i]) == Range(ops[i+1])
                @assert DDT(ops[i]) == RDT(ops[i+1])
            end
        end

        D = DDT(ops[N])
        R = RDT(ops[1])
        L = foldl(promote_linearity, map(linearity, ops))
        P = foldl(promote_parametricity, map(parametricity, ops))

        return new{D,R,L,P,typeof(ops),length(ops)}(ops)
    end
end

∘(ops::ParOperator...) = ParCompose(ops...)
∘(A::ParCompose, op::ParOperator) = ParCompose(A.ops..., op)
∘(op::ParOperator, A::ParCompose) = ParCompose(op, A.ops...)
∘(A::ParCompose, B::ParCompose) = ParCompose(A.ops..., B.ops...)
*(ops::ParLinearOperator...) = ∘(ops...)

Domain(A::ParCompose{D,R,L,P,F,N}) where {D,R,L,P,F,N} = Domain(A.ops[N])
Range(A::ParCompose{D,R,L,P,F,N}) where {D,R,L,P,F,N} = Range(A.ops[1])
children(A::ParCompose) = A.ops
rebuild(::ParCompose, cs) = ParCompose(cs...)

adjoint(A::ParCompose{D,R,Linear,P,F,N}) where {D,R,P,F,N} = ParCompose(reverse(map(adjoint, A.ops))...)

function (A::ParCompose{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractVector{D}}
    for i in 1:N
        # if state == 0
        #     println("Pre composition at op ", i, x)
        # end
        x = A.ops[N-i+1](x)
        # if state == 0
        #     println("Post composition at op ", i, x)
        # end
    end
    return x
end

function (A::ParCompose{D,R,L,<:Applicable,F,N})(x::X) where {D,R,L,F,N,X<:AbstractMatrix{D}}
    for i in 1:N
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        MPI.Barrier(MPI.COMM_WORLD)
        rank == 0 && println(typeof(A.ops[N-i+1]), size(x))
        x = A.ops[N-i+1](x)
    end
    return x
end

function *(x::X, A::ParCompose{D,R,Linear,<:Applicable,F,N}) where {D,R,F,N,X<:AbstractMatrix{R}}
    for i in 1:N
        x = x*A.ops[i]
    end
    return x
end

function latex_string(A::ParCompose{D,R,Linear,P,F,N}) where {D,R,P,F,N}
    child_eqns = [latex_string(c) for c in children(A)]
    if ast_location(A.ops[1]) == Internal
        out = "($(child_eqns[1]))"
    else
        out = child_eqns[1]
    end

    for i in 2:N
        if ast_location(A.ops[i]) == Internal
            out *= "\\ast($(child_eqns[i]))"
        else
            out *= "\\ast $(child_eqns[i])"
        end
    end
    return out
end

function latex_string(A::ParCompose{D,R,NonLinear,P,F,N}) where {D,R,P,F,N}
    child_eqns = [latex_string(c) for c in children(A)]
    if ast_location(A.ops[1]) == Internal
        out = "($(child_eqns[1]))"
    else
        out = child_eqns[1]
    end
    for i in 2:N
        if ast_location(A.ops[i]) == Internal
            out *= "\\circ($(child_eqns[i]))"
        else
            out *= "\\circ $(child_eqns[i])"
        end
    end
    return out
end

to_Dict(A::ParCompose) = Dict{String, Any}("type" => "ParCompose", "of" => map(to_Dict, A.ops))

function from_Dict(::Type{ParCompose}, d)
    ops = map(from_Dict, d["of"])
    ParCompose(ops...)
end

function transforms(A::ParCompose{D,R,L,P,F,N}) where {D,R,L,P,F,N}
    return Channel{ParOperator}() do channel
        found_any = false
        if length(A.ops) > 2
            # paren introduction
            # println("paren introduction")
            for first in range(1, length(A.ops)-1)
                for last in range(first+1, length(A.ops))
                    if first == 1 && last == length(A.ops)
                        # no point in putting parens around *all* of the ops, skip this transform
                        continue
                    end
                    ops = A.ops[first:last]
                    prefix = A.ops[1:first-1]
                    suffix = A.ops[last+1:length(A.ops)]
                    child = ParCompose(ops...)
                    transformed = ParCompose(prefix..., child, suffix...)
                    put!(channel, transformed)
                    found_any = true
                end
            end
        end
        for i in range(1, length(A.ops))
            op = A.ops[i]
            @match op begin
                child::ParCompose => begin
                    # paren destruction
                    # println("paren destruction")
                    prefix = A.ops[1:i-1]
                    suffix = A.ops[i+1:length(A.ops)]
                    transformed = ParCompose(prefix..., op.ops..., suffix...)
                    put!(channel, transformed)
                    found_any = true
                end
            end
        end

        if isa(L, Linear) && isa(F, AbstractVector{<:ParSeparableOperator}) && N == 2
            # Multiplication of two separable operators gives the ability to apply the mixed-product property.
            A_lhs = A.ops[1]
            A_rhs = A.ops[2]
            cs_lhs = children(A_lhs)
            cs_rhs = children(A_rhs)
            Nchild = length(cs_lhs)

            # If there are the same number of operators and they have matching dimensions,
            # it is valid to apply the rule
            if length(cs_rhs) == Nchild && all(DDT(l) == RDT(r) && Domain(l) == Range(r) for (l, r) in zip(cs_lhs, cs_rhs))
                for select in Iterators.product([[true, false] for _ in 1:Nchild]...)

                    # If no operators are selected from rhs, keep going
                    if !any(select)
                        continue
                    end

                    # Move operators from rhs -> lhs, replacing rhs w/ identity
                    ops_out_lhs = [select[i] ? cs_lhs[i]*cs_rhs[i] : cs_lhs[i] for i in 1:Nchild]
                    ops_out_rhs = [select[i] ? ParIdentity(DDT(cs_rhs[i]), Domain(cs_rhs[i])) : cs_rhs[i] for i in 1:Nchild]

                    # If we moved all operators, only return lhs, otherwise return
                    # combination of both
                    if all(select)
                        for op in transforms(rebuild(A_lhs, ops_out_lhs))
                            put!(channel, op)
                            found_any = true
                        end
                    else
                        lhs_out = rebuild(A_lhs, ops_out_lhs)
                        rhs_out = rebuild(A_rhs, ops_out_rhs)
                        for ops in Iterators.product(transforms(lhs_out), transforms(rhs_out))
                            put!(channel, ops[1]*ops[2])
                            found_any = true
                        end
                    end
                end
            end
        end

        # TODO: (AB = (B'A')')
    end
end
