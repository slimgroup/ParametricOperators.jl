export Optional, ParException

macro typeflag(typename, typevalues...)
    exprs = [:(abstract type $typename end)]
    for tv in typevalues
        push!(exprs, :(struct $tv <: $typename end))
    end
    return Expr(:block, exprs...)
end

const Optional{T} = Union{T, Nothing}
const ID = Union{UUID, String}

struct ParException <: Exception
    msg::String
end

struct LeftRight{T}
    left::T
    right::T
    LeftRight(left::T, right::T) where {T} = new{T}(left, right)
end

swap(lr::LeftRight) = LeftRight(lr.right, lr.left)