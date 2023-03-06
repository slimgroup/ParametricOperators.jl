export dot_string, plot_operator_graph, latex_string

simplename(x) = split("$(Base.typename(typeof(x)).wrapper)", ".")[end]

function dot_string(A::ParOperator, counts::AbstractDict; include_on_label=Dict())

    op_basename = simplename(A)
    counts[op_basename] += 1
    op_name = "$(op_basename)_$(counts[op_basename])"
    op_label = "$(op_basename) #$(counts[op_basename])"

    for (label_name, label_function) in pairs(include_on_label)
        try
            op_label *= "\n$label_name = $(label_function(A))"
        catch
            # ignored
        end
    end

    out = "\t$op_name [label=\"$op_label\"];\n"

    for c in children(A)
        (c_name, c_out) = dot_string(c, counts; include_on_label=include_on_label)
        out *= c_out
        out *= "\t$op_name -> $c_name;\n"
    end

    return (op_name, out)
end

function dot_string(A::ParOperator; include_on_label=Dict())
    counts = DefaultDict{String,Int}(0)
    graph_str = "digraph g {\n"
    (_, s) = dot_string(A, counts; include_on_label=include_on_label)
    graph_str *= s
    graph_str *= "}"
    return graph_str
end

function plot_operator_graph(A::ParOperator; name=nothing, ext="svg", include_on_label=Dict())
    name = isnothing(name) ? simplename(A) : name
    s = dot_string(A; include_on_label=include_on_label)
    run(pipeline(`echo $s`, `dot -T $ext -o $name.$ext`))
    return "$name.$ext"
end

latex_string(A::ParOperator) = "\\mathrm{$(simplename(A))}($(Domain(A)) \\rightarrow $(Range(A)))"