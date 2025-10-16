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

"""
    compute(infile::String, outfile::String)
"""
function compute(infile, outfile)
    # Read CSV file
    df = CSV.read(infile, DataFrame)
    # show(first(df, 5))

    # to have variables in rows and samples in columns in matrix, not adjoint
    D = Matrix{Int}(permutedims(Matrix(df), (2,1))) # rows: variables, columns: samples
    r = [maximum(df[!, col]) for col in names(df)]
    vars = [Variable(Symbol(var), r[i]) for (i, var) in enumerate(names(df))]

    ordering = collect(1:length(vars))
    # Ex: K2Ordering([1, 2, 3]). Here only var 1 can be parent of 2 and 3. 2 Can only be parent of 3, and so forth
    res = @timed K2_search(K2Ordering(ordering), vars, D)
    (k2score, G) = res.value
    println("K2 run time: $(round(res.time, digits=3)) s ") #,

    println("K2 DAG with $(nv(G)) nodes and $(ne(G)) edges")
    println("Score: ", k2score)
    vars_dict = Dict(i => vars[i].name for i in eachindex(vars))
    write_gph(G, vars_dict, k2score, "output/K2/", outfile)
    plot = gplot(G,
        nodelabel=(string(vars_dict[i]) for i in eachindex(vars)),
        nodelabeldist=1.5,          # distance from node center
        nodelabelsize=8,            # font size
        )

    draw(PNG("output/K2/" * outfile * ".png", 6inch, 6inch), plot)

    # run once, keep both result and timing
    res = @timed local_directed_graph_search(10000, vars, D)
    (ldgs_score, G) = res.value
    println("LDGS run time: $(round(res.time, digits=3)) s ") #,
            # "allocated $(Base.format_bytes(res.bytes))")

    println("Local DAG with $(nv(G)) nodes and $(ne(G)) edges")
    println("Score: ", ldgs_score)
    vars_dict = Dict(i => vars[i].name for i in eachindex(vars))
    write_gph(G, vars_dict, ldgs_score, "output/LDGS/", outfile)
    plot = gplot(G,
        nodelabel=(string(vars_dict[i]) for i in eachindex(vars)),
        nodelabeldist=1.5,          # distance from node center
        nodelabelsize=8,            # font size
        )
    draw(PNG("output/LDGS/" * outfile * ".png", 6inch, 6inch), plot)

    # TODO:
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
