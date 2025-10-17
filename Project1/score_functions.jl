using LinearAlgebra
using SpecialFunctions

struct Variable
    name::Symbol
    r::Int # cardinality
end

"""
    prior(n, r, q)
Compute the prior counts (Dirichlet parameters) for each variable in the Bayesian network.
    n: number of variables
    r: Vector of cardinalities for each variable
    q: Vector of number of parent configurations for each variable
    returns: Vector of prior counts matrices (q x r) for each variable
"""
function prior(n, r, q)
    return [ones(q[i], r[i]) for i in 1:n] # prior counts (Dirichlet alpha parameters)
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
    n: number of variables
    r: Vector of cardinalities for each variable
    q: Vector of number of parent configurations for each variable
    G: Directed graph (SimpleDiGraph)
    D: Data matrix (rows: variables, columns: observations)
    returns: Vector of statistics matrices M for each variable
"""
function statistics(n, r, q, G, D::Matrix{Int})
    # xi = size(D, 1) # number of variables (= nodes/rows in G)
    # n = size(D, 2) # number of observations/samples (= columns in D)

    # println("Computing statistics for $(xi) variables and $(size(D, 2)) observations")
    # r = [vars[i].r for i in 1:xi] # cardinalities
    # q = [isempty(inneighbors(G, i)) ? 1 : prod(r[j] for j in inneighbors(G, i)) for i in 1:xi] # parent configurations
    # for i in 1:xi
    #     println("Variable: ", vars[i].name, ", Cardinality: ", r[i], ", Parent configurations: ", q[i])
    # end
    M = [zeros(q[i], r[i]) for i in 1:n] # statistics matrices

    for o in eachcol(D) # for each observation
        for i in 1:n   # for each variable
            k = o[i]    # state of variable i in observation o
            parents = inneighbors(G, i)     # parents of variable i
            j = 1
            if !isempty(parents)    # if variable i has parents
                j = sub2ind(r[parents], o[parents])     # compute the index for the parent configuration
            end
            # println("Variable: ", vars[i].name, ", Parents: ", [vars[p].name for p in parents], ", Parent config index: ", j, ", State: ", k)
            M[i][j, k] += 1.0 # increment the count for this parent configuration and state
        end
    end
    return M
end

"""
    bayesian_score_component(M, alpha)
Compute the Bayesian score component for a single variable given its sufficient statistics and prior counts.
    M: Sufficient statistics matrix for the variable (q x r)
    alpha: Prior counts matrix for the variable (q x r)
    returns: Bayesian score component (Float64)
"""
function bayesian_score_component(M, alpha)
    p =  sum(loggamma.(alpha + M))
    p -= sum(loggamma.(alpha))
    p += sum(loggamma.(sum(alpha, dims=2)))
    p -= sum(loggamma.(sum(alpha, dims=2) + sum(M, dims=2)))
    return p
end

"""
    bayesian_score(vars, G, D)
Compute the total Bayesian score for a Bayesian network given the variables, graph structure, and data.
    vars: Vector of Variable structs
    G: Directed graph (SimpleDiGraph)
    D: Data matrix (rows: variables, columns: observations)
    returns: Total Bayesian score (Float64)
"""
function bayesian_score(vars, G::SimpleDiGraph, D)
    n = length(vars)
    r = [vars[i].r for i in 1:n] # cardinalities
    q = [isempty(inneighbors(G, i)) ? 1 : prod(r[j] for j in inneighbors(G, i)) for i in 1:n] # parent configurations

    M = statistics(n, r, q, G, D)
    alpha = prior(n, r, q)

    return sum(bayesian_score_component(M[i], alpha[i]) for i in 1:n)
end
