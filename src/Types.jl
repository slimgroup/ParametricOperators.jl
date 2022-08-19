"""
Package level exception type.
"""
struct TaoException
    msg::String
end

"""
Optional type holding either a T or nothing.
"""
Optional{T} = Union{T, Nothing}

"""
Alias for array shape.
"""
Shape = Union{Vector{Int64},Tuple{Vararg{Int64}}}