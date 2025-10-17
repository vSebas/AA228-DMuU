using GraphPlot, Graphs, Compose, Cairo
using Printf
using CSV
using DataFrames
using BenchmarkTools

include("search_algos.jl")

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
Also writes the score in a separate `.score` file.
"""
function write_gph(dag::DiGraph, idx2names, score, path, filename)
    isdir(path) || mkdir(path)

    open(path * filename * ".gph", "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end

    open(path * filename * ".score", "w") do io
        @printf(io, "%f\n", score)
    end
end

function search(vars, D)
    ordering = collect(1:length(vars))
    __, G = K2_search(K2Ordering(ordering), vars, D)
    return local_directed_graph_search(LDGS(G, 100000), vars, D)
end

"""
    compute(infile::String, outfile::String)
"""
function compute(infile, outfile)
    # Read CSV file
    df = CSV.read(infile, DataFrame)

    D = Matrix{Int}(permutedims(Matrix(df), (2,1))) # rows: variables, columns: samples
    r = [maximum(df[!, col]) for col in names(df)]
    vars = [Variable(Symbol(var), r[i]) for (i, var) in enumerate(names(df))]

    res = @timed search(vars, D)
    (score, G) = res.value
    println("Run time: $(round(res.time, digits=3)) s ")
    println("Directed Acyclic Graph with $(nv(G)) nodes and $(ne(G)) edges")
    println("Score: ", score)
    vars_dict = Dict(i => vars[i].name for i in eachindex(vars))
    write_gph(G, vars_dict, score, "output/combined/", outfile)
    plot = gplot(G, nodelabel=(string(vars_dict[i]) for i in eachindex(vars)), arrowlengthfrac=0.07)
    draw(PNG("output/combined/" * outfile * ".png", 6inch, 6inch), plot)

    # improvements:
    # TRY DIFFERENT ORDERINGS FOR K2 (RANDOM, REVERSED, ETC)
    # INCREASE K FOR LDGS
    # TRY DIFFERENT INITIAL GRAPHS FOR LDGS (EMPTY, FULL, RANDOM)
    # TRY DIFFERENT STRATEGIES TO AVOID LOCAL OPTIMA IN LDGS (SIMULATED ANNEALING, TABU SEARCH, ETC)

end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)
