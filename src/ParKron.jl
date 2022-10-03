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

function (K::ParKron{D,R,<:Applicable,F})(x::X) where {D,R,F,X<:AbstractVector{D}}
    y = reshape(x, K.shape_in...)
    N = length(size(y))
    for i in K.order
        Ai = K.ops[i]
        y = mapslices(Ai, y, dims=N-i+1)
    end
    return vec(y)
end

function kron_pullback(K::ParKron{D,R,<:Applicable,F}, x::X, config::RuleConfig{>:HasReverseMode}) where {D,R,F,X<:AbstractVector{D}}
    y = reshape(x, K.shape_in...)
    N = length(size(y))
    for i in K.order
        Ai = K.ops[i]
        rule = nothing
        y = mapslices(sl -> begin 
            rule = isnothing(rule) ? rrule_via_ad(config, Ai, sl) : rule
            _, out = rule(sl)
            return out
        end, y, dims=N-i+1)
    end
    return vec(y)
end

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, K::ParKron{D,R,Parameterized,F}, x::X) where {D,R,F,X<:AbstractVecOrMat{D}}
    y = K(x)
    function pullback(Δy::Y) where {R,Y<:AbstractVector{R}}
        NoTangent(), @thunk(kron_pullback(K', Δy, config))
    end
    y, pullback
end