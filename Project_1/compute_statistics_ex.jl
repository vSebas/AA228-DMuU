using Graphs
using Distributions
using LinearAlgebra

struct Variable
    name::Symbol
    r::Int # cardinality
end

const Assignment = Dict{Symbol, Int}
const FactorTable = Dict{Assignment, Float64}

struct Factor
    vars::Vector{Variable}
    table::FactorTable
end

"""
    sub2ind(siz, x)
Convert subscript indices `x` to a linear index for an array of size `siz`.
    siz: Vector of sizes for each dimension
    x: Vector of subscript indices (1-based)
    returns: linear index (1-based)
"""

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1])) # cumulative product of sizes
    return dot(k, x .- 1) + 1
end

"""
    statistics(vars, G, D)
Compute the sufficient statistics for a Bayesian network given the variables, graph structure, and data.
    vars: Vector of Variable structs
    G: Directed graph (SimpleDiGraph)
    D: Data matrix (rows: variables, columns: observations)
    returns: Vector of statistics matrices M for each variable
"""

function statistics(vars, G, D::Matrix{Int})
    xi = size(D, 1) # number of variables (= nodes in G)
    n = size(D, 2) # number of observations/samples (= columns in D)

    r = [vars[i].r for i in 1:xi] # cardinalities
    q = [isempty(inneighbors(G, i)) ? 1 : prod(r[j] for j in inneighbors(G, i)) for i in 1:xi] # parent configurations

    # inneighbors(G, i) gives the parents of node i in the graph G by returning the nodes with 
    # edges directed to i that is the parents of i in a Bayesian network

    M = [zeros(q[i], r[i]) for i in 1:xi] # statistics matrices

    for o in eachcol(D) # for each observation
        for i in 1:xi   # for each variable
            k = o[i]    # state of variable i in observation o
            parents = inneighbors(G, i)     # parents of variable i
            j = 1
            if !isempty(parents)    # if variable i has parents
                j = sub2ind(r[parents], o[parents])     # compute the index for the parent configuration
            end
            M[i][j, k] += 1.0 # increment the count for this parent configuration and state
        end
    end
    return M
end

G = SimpleDiGraph(5)    
add_edge!(G, 1, 2)
add_edge!(G, 1, 3)

vars = [Variable(:A, 2), Variable(:B, 2), Variable(:C, 2)]
D = [1 2 2 1; 1 2 2 1; 2 2 2 2]
# 3 variables, 4 observations
# A B C
# 1 1 2
# 2 2 2
# 2 2 2
# 1 1 2
# A is the parent of B and C

M = statistics(vars, G, D)

println(M)