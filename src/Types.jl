Optional{T} = Union{T, Nothing}
Shape = Tuple{Vararg{Int64}}

struct TaoException
    msg::String
end