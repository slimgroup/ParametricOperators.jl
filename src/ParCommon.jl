export optional_vcat

optional_vcat(::Nothing, ::Nothing) = nothing
optional_vcat(x, ::Nothing) = x
optional_vcat(::Nothing, x) = x
optional_vcat(x, y) = vcat(x, y)

const Option{T} = Union{T, Nothing}
const ID = Union{UUID, String}

struct ParException
    msg::String
end