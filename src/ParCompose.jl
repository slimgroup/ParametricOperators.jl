export ParCompose

abstract type ParamOrder end
struct LeftFirst <: ParamOrder end
struct RightFirst <: ParamOrder end

struct ParCompose{D,R,L,P,F,O<:ParamOrder} <: ParOperator{D,R,L,P,HigherOrder}
    ops::F
    m::Int64
    n::Int64
    ranges::Vector{Option{UnitRange{Int64}}}
    id::ID

    function ParCompose(ops; param_order=LeftFirst)
        
        D = DDT(ops[end])
        R = RDT(ops[1])
        L = foldr(promote_linearity, map(linearity, ops); init=Linear)
        P = foldr(promote_parametricity, map(parametricity, ops); init=NonParametric)

        # Check dimension and type compatibility
        N = length(ops)
        for i ∈ 1:N-1
            @assert DDT(ops[i]) == RDT(ops[i+1])
            @assert Domain(ops[i]) == Range(ops[i+1])
        end

        m = Range(ops[1])
        n = Domain(ops[end])

        nps = collect(map(nparams, ops))
        offsets = [0, cumsum(nps[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ nps
        ranges = [start:stop for (start, stop) in zip(starts, stops)]
        ranges = collect(map(r -> length(r) == 0 ? nothing : r, ranges))

        return new{D,R,L,P,typeof(ops),param_order}(ops, m, n, ranges, uuid4(GLOBAL_RNG))
    end
end

∘(ops::ParOperator...) = ParCompose(collect(ops))
*(ops::ParLinearOperator...) = ParCompose(collect(ops))

Domain(A::ParCompose) = A.n
Range(A::ParCompose) = A.m
children(A::ParCompose) = A.ops
id(A::ParCompose) = A.id

adjoint(A::ParCompose{D,R,Linear,P,F,LeftFirst}) where {D,R,P,F} = ParCompose(collect(map(adjoint, reverse(A.ops))); param_order=RightFirst)
adjoint(A::ParCompose{D,R,Linear,P,F,RightFirst}) where {D,R,P,F} = ParCompose(collect(map(adjoint, reverse(A.ops))); param_order=LeftFirst)

function (A::ParCompose{D,R,L,Parametric,F,LeftFirst})(θ::V) where {D,R,L,F,V}
    ops_out = [isnothing(r) ? op : op(view(θ, r)) for (op, r) in zip(A.ops, A.ranges)]
    return ParCompose(ops_out)
end

function (A::ParCompose{D,R,L,Parametric,F,RightFirst})(θ::V) where {D,R,L,F,V}
    N = length(θ)
    ops_out = [isnothing(r) ? op : op(view(θ, N-r.stop+1:N-r.stop+length(r))) for (op, r) in zip(A.ops, A.ranges)]
    return ParCompose(ops_out)
end

function (A::ParCompose{D,R,L,P,F,O})(x::X) where {D,R,L,P<:Applicable,F,O,X<:AbstractVector{D}}
    y = x
    for op in reverse(A.ops)
        y = op(y)
    end
    return y
end

function (A::ParCompose{D,R,L,P,F,O})(x::X) where {D,R,L,P<:Applicable,F,O,X<:AbstractMatrix{D}}
    y = x
    for op in reverse(A.ops)
        y = op(y)
    end
    return y
end