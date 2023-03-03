export to_json, from_json

import JSON

function to_json(A::ParOperator)
    s = to_Dict(A)
    JSON.json(s)
end

function from_json(s::String)
    d = JSON.parse(s)
    from_Dict(d)
end

ParOperator_TYPES = Dict{String, Any}(
    "ParAdjoint"     => ParAdjoint,
    # "ParBroadcasted" => ParBroadcasted,
    "ParCompose"     => ParCompose,
    "ParDFT"         => ParDFT,
    "ParDiagonal"    => ParDiagonal,
    # "ParDistributed" => ParDistributed,
    "ParIdentity"    => ParIdentity,
    "ParKron"        => ParKron,
    "ParMatrix"      => ParMatrix,
    # "ParRepartition" => ParRepartition,
    "ParRestriction" => ParRestriction,
)

Data_TYPES = Dict{String, Any}(
    "Float16" => Float16,
    "Float32" => Float32,
    "Float64" => Float64,
    "ComplexF16" => ComplexF16,
    "ComplexF32" => ComplexF32,
    "ComplexF64" => ComplexF64,
)

function from_Dict(d::Dict{String, Any})
    if ! haskey(d, "type")
        throw(ParException("dict has no 'type' key"))
    end

    ts = d["type"]
    if ! haskey(ParOperator_TYPES, ts)
        throw(ParException("cannot convert Dict to unknown type `$ts`"))
    end

    op = ParOperator_TYPES[ts]

    from_Dict(op, d)
end
