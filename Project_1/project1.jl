using Graphs
using Printf
using CSV, DataFrames

include("search_algos.jl")

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
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
    r = [length(unique(df[!, col])) for col in names(df)]

    vars = [Variable(Symbol(var), r[i]) for (i, var) in enumerate(names(df))]

    ordering = collect(1:length(vars))
    score, G = fit(K2Search(ordering), vars, D)

    println("DAG with $(nv(G)) nodes and $(ne(G)) edges")
    println("Score: ", score)
    write_gph(G, Dict(i => vars[i].name for i in eachindex(vars)), outfile)
end

if length(ARGS) != 2
    error("usage: julia project1.jl <infile>.csv <outfile>.gph")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]

compute(inputfilename, outputfilename)