export ParKron, ⊗

struct ParKron{D,R,P,F} <: ParLinearOperator{D,R,P,Internal}
    ops::F
    m::Int64
    n::Int64
    shape_in::Vector{Int64}
    shape_out::Vector{Int64}
    order::Vector{Int64}
    ranges::Vector{UnitRange{Int64}}
    slots::Set{Int64}
    id::ID
    function ParKron(ops::ParLinearOperator{<:Number,R,<:Parametricity,<:ASTLocation}...) where {R}
        shape_in = reverse(collect(map(Domain, ops)))
        shape_out = reverse(collect(map(Range, ops)))
        n = prod(shape_in)
        m = prod(shape_out)

        D_out = DDT(ops[end])
        order = Vector{Int64}()
        for (i, op) in reverse(collect(enumerate(ops)))
            D_op = DDT(op)
            if D_op == D_out
                push!(order, i)
            else
                T = promote_type(D_op, D_out)
                if T == D_out
                    D_out = D_op
                    pushfirst!(order, i)
                else
                    push!(order, i)
                end
            end
        end

        P_out = foldl((p1, p2) -> promote_parametricity(p1, p2), map(parametricity, ops); init = NonParametric)
        offsets = [0, cumsum(map(nparams, ops[1:end-1]))...]
        starts = offsets .+ 1
        stops = [o+np for (o, np) in zip(offsets, map(nparams, ops))]
        ranges = [start:stop for (start, stop) in zip(starts, stops)]
        slots = Set(map(tup -> tup[1], filter(tup -> length(tup[2]) > 0, collect(enumerate(ranges)))))
        return new{D_out,R,P_out,typeof(ops)}(ops, m, n, shape_in, shape_out, order, ranges, slots, uuid4(GLOBAL_RNG))
    end
    function ParKron(D, R, P, ops, m, n, si, so, order, ranges, slots, id)
        return new{D,R,P,typeof(ops)}(ops, m, n, si, so, order, ranges, slots, id)
    end
end

kron(lhs::ParLinearOperator, rhs::ParLinearOperator) = ParKron(lhs, rhs)
kron(lhs::ParLinearOperator, rhs::ParKron) = ParKron(lhs, rhs.ops...)
kron(lhs::ParKron, rhs::ParLinearOperator) = ParKron(lhs.ops..., rhs)
kron(lhs::ParKron, rhs::ParKron) = ParKron(lhs.ops..., rhs.ops...)

⊗(ops::ParLinearOperator...) = kron(ops...)

Domain(A::ParKron) = A.n
Range(A::ParKron) = A.m
children(A::ParKron) = A.ops
id(A::ParKron) = A.id
function adjoint(A::ParKron{D,R,P,F}) where {D,R,P,F}
    ops_adj = Tuple(map(adjoint, A.ops))
    return ParKron(
        R,
        D,
        P,
        ops_adj,
        A.n,
        A.m,
        A.shape_out,
        A.shape_in,
        reverse(A.order),
        A.ranges,
        A.slots,
        "adjoint_[$(A.id)]"
    )
end

(A::ParKron{D,R,Parametric,F})(θ::AbstractVector{<:Number}) where {D,R,F} =
    ParKron([i ∈ A.slots ? op(θ[range]) : op for (i, (op, range)) in enumerate(zip(A.ops, A.ranges))]...)

function (K::ParKron{D,R,<:Applicable,F})(x::X; config::Optional{RuleConfig} = nothing) where {D,R,F,X<:AbstractVector{D}}

    # Compute the intermediate shapes and types from applying K in its
    # given order
    shapes = [K.shape_in]
    types = [DDT(K.ops[K.order[1]])]
    N = length(K.shape_in)
    for (k, i) in enumerate(K.order)
        Ai = K.ops[i]
        s = copy(shapes[k])
        s[N-i+1] = Range(Ai)
        push!(shapes, s)
        push!(types, RDT(Ai))
    end

    # Allocate left/right buffers for application of each operator in K. These
    # buffers are stored as raw bytes to allow for datatype reinterpretation at
    # runtime.
    bytes = [sizeof(T)*prod(s) for (T, s) in zip(types, shapes)]
    max_bytes = maximum(bytes)
    vecs = LeftRight(zeros(UInt8, max_bytes), zeros(UInt8, max_bytes))
    copyto!(vecs.right, reinterpret(UInt8, x))
    
    for (k, i) in enumerate(K.order)

        # Swap left/right pointers
        vecs = swap(vecs)

        s_in  = shapes[k]
        s_out = shapes[k+1]
        T_in  = types[k]
        T_out = types[k+1]

        l_buf = @view reinterpret(T_in, vecs.left)[1:prod(s_in)]
        r_buf = @view reinterpret(T_out, vecs.right)[1:prod(s_out)]
        l_buf = reshape(l_buf, s_in...)
        r_buf = reshape(r_buf, s_out...)

        # Create cartesian index iterators for dimensions above/below d
        # (the dimension over which Aᵢ is applied)
        d = N-i+1
        idxs_lower_in  = CartesianIndices(Tuple(s_in[1:d-1]))
        idxs_upper_in  = CartesianIndices(Tuple(s_in[d+1:N]))
        idxs_lower_out = CartesianIndices(Tuple(s_out[1:d-1]))
        idxs_upper_out = CartesianIndices(Tuple(s_out[d+1:N]))
        
        # Non-allocating variant of mapslices() that uses the above left/right
        # data structure.
        if isnothing(config)
            Ai = K.ops[i]
            for (li, lo) in zip(idxs_lower_in, idxs_lower_out)
                for (ui, uo) in zip(idxs_upper_in, idxs_upper_out)
                    r_buf[lo,:,uo] = Ai*l_buf[li,:,ui]
                end
            end
        else
            for (li, lo) in zip(idxs_lower_in, idxs_lower_out)
                for (ui, uo) in zip(idxs_upper_in, idxs_upper_out)
                    rule = rrule_via_ad(config, Ai, *, l_buf[li,:,ui])
                    _, r_buf[lo,:,uo] = rule(l_buf[li,:,ui])
                end
            end
        end
    end

    out = reinterpret(R, vecs.right)
    return @view out[1:Range(K)]
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, K::ParKron{D,R,Parameterized,F}, x::X) where {D,R,F,X<:AbstractVector{D}}
    y = K(x)
    function pullback(Δy::Y) where {R,Y<:AbstractVector{R}}
        NoTangent(), @thunk(K'(Δy; config = config))
    end
    y, pullback
end