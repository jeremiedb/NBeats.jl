module NBeats

using Flux
using Flux:@functor

export NBeatsBlock, NBeatsStack

"""
    NBeatsBlock: funcdamental building block of an NBeats model.
"""
struct NBeatsBlock
    core::Vector
    back_fc::Dense
    forw_fc::Dense
    back_scale::Vector
    forw_scale::Vector
end

@functor NBeatsBlock

function NBeatsBlock(; back_size, forw_size, num_hidden, num_layers_core)
    core = []
    push!(core, Dense(back_size, num_hidden, relu))
    for i in 2:num_layers_core
        push!(core, Dense(num_hidden, num_hidden, relu))
    end

    m = NBeatsBlock(
            core,
            Dense(num_hidden, back_size),
            Dense(num_hidden, forw_size),
            ones(Float32, back_size),
            ones(Float32, forw_size))
    return m
end

function (m::NBeatsBlock)(x)
    core = foldl((x, l) -> l(x), m.core, init=x)
    back = m.back_scale .* m.back_fc(core)
    forw = m.forw_scale .* m.forw_fc(core)
    return back, forw
end

"""
    NBeatsStack: structure containing a vector of `NBeatsBlock`
"""
struct NBeatsStack
    blocks::Vector
end

@functor NBeatsStack

function NBeatsStack(; back_size, forw_size, num_hidden, num_layers_core, num_blocks)
    blocks = []
    for i in 1:num_blocks
        push!(blocks, NBeatsBlock(;back_size=back_size, forw_size=forw_size, num_hidden=num_hidden, num_layers_core=num_layers_core))
    end
    m = NBeatsStack(blocks)
    return m
end

function (m::NBeatsStack)(x)
    block = m.blocks[1](x)
    x = x .- block[1]
    out = block[2]
    for i in 2:length(m.blocks) - 1
        block = m.blocks[i](x)
        x = x .- block[1]
        out = out .+ block[2]
    end
    block = m.blocks[length(m.blocks)](x)
    out = out .+ block[2]
    return out
end

end # module
