using ChainRulesCore
using Flux
using Functors
using OMEinsum

combine_branch_trunk(b::AbstractMatrix, t::AbstractMatrix) = ein"sb,sb->b"(b, t)
combine_branch_trunk(b::AbstractMatrix, t::AbstractArray{T,3}) where {T} = ein"sb,snb->nb"(b, t)

struct Bias
    params::AbstractArray
    Bias(params::AbstractArray) = new(params)
    Bias(T::Type, dims::Int...) = new(zeros(T, dims...))
    Bias(dims::Int...) = Bias(Float32, dims...)
end

@functor Bias

(B::Bias)(x) = x .+ B.params

DeepONet(branch_net, trunk_net, out_net) =
    Chain(
        Parallel(
            combine_branch_trunk,
            branch_net,
            trunk_net
        ),
        out_net
    )

atleast_2d(x::AbstractVector) = reshape(x, length(x), 1)
atleast_2d(x::AbstractArray) = x

function sample(u::AbstractArray, locations::AbstractMatrix{<:Integer})
    s = size(u)
    b = s[end]
    out = mapreduce(
        arr -> mapreduce(
            loc -> arr[loc...],
            vcat,
            eachcol(locations)
        ),
        hcat,
        view(u, [1:d for d in s[1:end-1]]..., i) for i in 1:b
    )
    return atleast_2d(out)
end

function sample(u::AbstractArray, locations::AbstractArray{<:Integer,3})
    s = size(u)
    b = s[end]
    n_locations = size(locations)[2]
    samples = mapreduce(i -> sample(view(u, [1:d for d in s[1:end-1]]..., i:i), locations[:,:,i]), hcat, 1:b)
    return reshape(samples, n_locations, b)
end

function dense_branch_net(n_sensor_locations, hidden_scale = 2, n_layers = 4, act = gelu)
    n_hidden = n_sensor_locations*hidden_scale
    
    layers = [Chain(
        Dense(n_sensor_locations => n_hidden, act),
        Dense(n_hidden => n_sensor_locations, i == n_layers ? identity : act),
    ) for i in 1:n_layers]

    return Chain(
        layers...
    )

end

function swapdims(x, d0, d1)
    n = length(size(x))
    perm = collect(1:n)
    @ignore_derivatives perm[d0], perm[d1] = perm[d1], perm[d0]
    return permutedims(x, perm)
end

function dense_trunk_net(loc_dim, n_sensor_locations, hidden_scale = 2, n_layers = 4, act = gelu)

    n_hidden = hidden_scale*n_sensor_locations
    layers = [Chain(
        Dense(n_sensor_locations => n_hidden, act),
        Dense(n_hidden => n_sensor_locations, i == n_layers ? identity : act),
    ) for i in 1:n_layers]

    return Chain(
        Dense(loc_dim, n_sensor_locations, act),
        layers...
    )

end