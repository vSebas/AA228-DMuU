using Graphs
using Printf
using CSV, DataFrames

include("search_algos.jl")

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
Also writes the score in a separate `.score` file.
"""
function write_gph(dag::DiGraph, idx2names, score, filename)
    open(filename * ".gph", "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end

    open(filename * ".score", "w") do io
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
    # r = [length(unique(df[!, col])) for col in names(df)] # this one is wrong if some value is missing in the data. Unique counts the number of distinct values, not the range of values
    vars = [Variable(Symbol(var), r[i]) for (i, var) in enumerate(names(df))]

    ordering = collect(1:length(vars))
    # Ex: K2Search([1, 2, 3]). Here only var 1 can be parent of 2 and 3. 2 Can only be parent of 3, and so forth
    score, G = fit(K2Search(ordering), vars, D)

    println("DAG with $(nv(G)) nodes and $(ne(G)) edges")
    println("Score: ", score)
    write_gph(G, Dict(i => vars[i].name for i in eachindex(vars)), score, outfile)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)