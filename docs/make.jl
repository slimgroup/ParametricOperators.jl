using Pkg
Pkg.activate("../")

using ParametricOperators
using Documenter

push!(LOAD_PATH, "../src")
makedocs(
    sitename="ParametricOperators.jl",
    modules=[ParametricOperators],
    pages=[
        "Home" => "index.md",
        "Examples" => "examples.md"
    ],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true"
    ))
deploydocs(;
    repo="github.com/slimgroup/ParametricOperators.jl"
)
