export ParSeparableOperator, ParKron, ⊗

abstract type ParSeparableOperator{D,R,P,O} <: ParLinearOperator{D,R,P,O} end

decomposition(A::ParSeparableOperator{D,R,P,HigherOrder}) where {D,R,P} = children(A)

subspace_type(::Type{T}, ::Type{T}) where {T} = T
subspace_type(::Type{T}, ::Type{Complex{T}}) where {T<:Real} = T
subspace_type(::Type{Complex{T}}, ::Type{T}) where {T<:Real} = T

mutable struct ParKron{D,R,P,F,N} <: ParSeparableOperator{D,R,P,HigherOrder}
    ops::F
    m::Int64
    n::Int64
    order::Vector{Int64}
    perms::Vector{NTuple{N, Int64}}
    shapes::Vector{NTuple{N, Int64}}
    ranges::Vector{Option{UnitRange{Int64}}}
    buf::Option{<:AbstractVector{UInt8}}
    id::ID

    function ParKron(ops)

        N = length(ops)

        # Compute the order of application of the given operators by finding
        # an order in which a chain of types
        #
        # T₁ ⊆ T₂ ⊆ … ⊆ Tₙ
        #
        # is formed.
        Ds = collect(map(DDT, ops))
        Rs = collect(map(RDT, ops))
        T  = reduce(subspace_type, Ds)
        order = zeros(Int64, N)
        for i ∈ 1:N
            js = filter(j -> j ∉ order, findall(Ds .== T))
            Rs_i = Rs[js]
            R = reduce(subspace_type, Rs_i)
            j = js[findlast(Rs_i .== R)]
            order[i] = j
        end

        # From the order, compute intermediate shapes
        shifts = zeros(Int64, N+1)
        shifts[1] = order[1]-N
        for i ∈ 1:N-1
            shifts[i+1] = order[i+1] - order[i]
        end
        shifts[N+1] = N-order[N]

        shapes = Vector{NTuple{N, Int64}}(undef, N+1)
        shapes[1] = ntuple(j -> Domain(ops[N-j+1]), N)

        for i ∈ 1:N
            o = order[i]
            shape = collect(map(j -> j == 1 ? Range(ops[o]) : shapes[i][j], 1:N))
            shapes[i+1] = Tuple(circshift(shape, shifts[i+1]))
        end

        # Convert shifts to permutations
        perms = collect(map(s -> Tuple(circshift(1:N, s)), shifts))
        
        D = Ds[order[1]]
        R = Rs[order[N]]
        P = foldr(promote_parametricity, map(parametricity, ops); init=NonParametric)

        m = prod(shapes[N+1])
        n = prod(shapes[1])

        nps = collect(map(nparams, ops))
        offsets = [0, cumsum(nps[1:end-1])...]
        starts = offsets .+ 1
        stops = offsets .+ nps
        ranges = [start:stop for (start, stop) in zip(starts, stops)]
        ranges = collect(map(r -> length(r) == 0 ? nothing : r, ranges))

        max_bytes = 0
        for i ∈ 1:N
            b1 = prod(shapes[i])*sizeof(DDT(ops[order[i]]))
            b2 = prod(shapes[i+1])*sizeof(RDT(ops[order[i]]))
            max_bytes = max(max_bytes, b1, b2)
        end
        buf = nothing #zeros(UInt8, max_bytes)

        return new{D,R,P,typeof(ops),N}(ops, m, n, order, perms, shapes, ranges, buf, uuid4(GLOBAL_RNG))
    end

    ParKron(D,R,P,ops,m,n,order,perms,shapes,ranges,buf,id) =
        new{D,R,P,typeof(ops),length(ops)}(ops,m,n,order,perms,shapes,ranges,buf,id)
end

kron(A::ParLinearOperator, B::ParLinearOperator) = ParKron([A, B])
kron(A::ParLinearOperator, B::ParSeparableOperator) = ParKron(vcat(A, decomposition(B)))
kron(A::ParSeparableOperator, B::ParLinearOperator) = ParKron(vcat(decomposition(A), B))
kron(A::ParSeparableOperator, B::ParSeparableOperator) = ParKron(vcat(decomposition(A), decomposition(B)))

⊗(A::ParLinearOperator, B::ParLinearOperator) = kron(A, B)

Domain(A::ParKron) = A.n
Range(A::ParKron) = A.m
children(A::ParKron) = A.ops
id(A::ParKron) = A.id

function (A::ParKron{D,R,Parametric,F,N})(θ::V) where {D,R,F,N,V}
    ops_out = [isnothing(r) ? op : op(view(θ, r)) for (op, r) in zip(A.ops, A.ranges)]
    return ParKron(D,R,Parameterized,ops_out,A.m,A.n,A.order,A.perms,A.shapes,A.ranges,A.buf,A.id)
end

function permutedims_noalloc!(dest::U, src::V, perm) where {T,U<:AbstractArray{T},V<:AbstractArray{T}}
    permutedims!(dest, src, perm)
    return dest
end

function ChainRulesCore.rrule(::typeof(permutedims_noalloc!), dest::U, src::V, perm) where {T,U<:AbstractArray{T},V<:AbstractArray{T}}
    y = permutedims_noalloc!(dest, src, perm)
    function pullback(∂y)
        N = length(perm)
        permᵀ = zeros(Int64, N)
        for i ∈ 1:N
            permᵀ[perm[i]] = i
        end
        ∂x = permutedims_noalloc!(src, ∂y, permᵀ)
        return NoTangent(), NoTangent(), ∂x, NoTangent()
    end
    return y, pullback
end

function (A::ParKron{D,R,P,F,N})(x::X) where {D,R,P<:Applicable,F,N,X<:AbstractVector{D}}

    y = reshape(x, map(Domain, reverse(A.ops))...)
    for i ∈ 1:N
        o = A.order[i]
        Ai = A.ops[o]
        #dest_i = reshape(view(reinterpret(DDT(Ai), A.buf), 1:length(y)), A.shapes[i])
        #y = permutedims_noalloc!(dest_i, y, A.perms[i]) #TODO: make this work
        y = permutedims(y, A.perms[i])
        s = size(y)
        y = reshape(y, Domain(Ai), length(y)÷Domain(Ai))
        y = Ai*y
        y = reshape(y, Range(Ai), s[2:end]...)
    end
    #dest_i = reshape(view(reinterpret(RDT(A.ops[A.order[N]]), A.buf), 1:length(y)), A.shapes[N+1])
    #y = permutedims_noalloc!(dest_i, y, A.perms[N+1])
    y = permutedims(y, A.perms[N+1])
    return vec(copy(y))

end