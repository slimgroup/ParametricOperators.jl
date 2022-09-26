export Optional, TaoException

macro typeflag(typename, typevalues...)
    exprs = [:(abstract type $typename end)]
    for tv in typevalues
        push!(exprs, :(struct $tv <: $typename end))
    end
    return Expr(:block, exprs...)
end

const Optional{T} = Union{T, Nothing}
const ID = Union{UUID, String}

struct TaoException <: Exception
    msg::String
end

dims_compatible(::Nothing, ::Nothing) = true
dims_compatible(::Nothing, ::Integer) = true
dims_compatible(::Integer, ::Nothing) = true
dims_compatible(d1::Integer, d2::Integer) = d1 == d2