using ParametricOperators

A = ParMatrix(Float64, 3, 3, "A")
B = ParMatrix(Float64, 3, 3, "B")
C = ParMatrix(Float64, 3, 3, "C")
D = ParMatrix(Float64, 3, 3, "D")
E = ParMatrix(Float64, 3, 3, "E")
F = ParMatrix(Float64, 3, 3, "F")
G = ParMatrix(Float64, 3, 3, "G")

expr = ParCompose(A,ParCompose(B,ParCompose(C,ParCompose(D,ParCompose(E,ParCompose(F,G))))))

orig_fname = plot_operator_graph(expr; name="orig", include_on_label=Dict("Domain" => Domain, "Range" => Range, "order" => ParametricOperators.order, "id" => c -> c.id))
println("orig: $orig_fname")

opt = Optimizer(expr)
while step_optimizer(opt)
    println("working set is $(length(opt.best)) elements.  seen $(opt.num_total_seen), discarded $(opt.num_dups) duplicates.")
end
println("working set is $(length(opt.best)) elements.  seen $(opt.num_total_seen), discarded $(opt.num_dups) duplicates.")
best = opt.best[1][3]

println("best is $best")

best_fname = plot_operator_graph(best; name="best", include_on_label=Dict("Domain" => Domain, "Range" => Range, "order" => ParametricOperators.order, "id" => c -> c.id))
println("best: $best_fname")
