

"""
Plot a histogram of sampling and integration elapsed CPU times.
"""
function plot_time_histogram(samples_ps::Union{T,MCMC_Samples}; size=(8,5)) where {T<:NamedTuple}
    
    max_t = maximum(maximum.([samples_ps.time_mcmc, samples_ps.time_integration]))
    bins=collect(0:1:floor(max_t))
    
    fig, ax = plt.subplots(1,1, figsize=size)
    
    ax.hist(samples_ps.time_mcmc, bins, color="royalblue", label = "Sampling Time", alpha=0.8);
    ax.hist(samples_ps.time_integration, bins, color="peru", label = "Integration Time", alpha=0.8);

    ax.legend(loc="upper right", frameon=false, framealpha=0.8, ncol=2, )
    ax.set_xlabel("Time [s]", labelpad=10,  size=12)
    ax.set_ylabel("Counts", labelpad=10,  size=12)
end



"""
Plot overlapping timeline of wall-clock times from different workers. 
"""
function plot_overlapped_timeline(samples_ps::Union{T,MCMC_Samples}; size=(8,5)) where {T<:NamedTuple}
    
    minimum_timestamp = minimum(minimum(samples_ps.timestamps))
    timestamps_shifted = [(i .- minimum_timestamp)*10^-9  for i in samples_ps.timestamps]
    maximum_timestamp  = maximum(maximum(timestamps_shifted))
    maximum_cpu_time = maximum([samples_ps.time_mcmc..., samples_ps.time_integration...])
    
    proc_ids = sort(unique(samples_ps.proc_id))
    n_workers = length(proc_ids)
    
    x_min = 0
    x_max = maximum_timestamp
    y_min = 0
    y_max_tmp = Float64[]
    
    label_1 = "Sampling"
    label_2 = "Integration"
    
    fig, ax = plt.subplots(1,1, figsize=size)

    for (i,j) in enumerate(timestamps_shifted)
    
        r1 = matplotlib.patches.Rectangle([j[1], 0.0], j[2]-j[1], samples_ps.time_mcmc[i]/(j[2]-j[1]), 
            fill=false, linewidth=1, color="royalblue", alpha=1, label=label_1) 
        
        ax.add_patch(r1)

        r2 = matplotlib.patches.Rectangle([j[3], 0.0], j[4]-j[3], samples_ps.time_integration[i]/(j[4]-j[3]), 
            fill=false, linewidth=1, color="peru", alpha=1, label=label_2) 
        
        ax.add_patch(r2)

        push!(y_max_tmp, samples_ps.time_mcmc[i]/(j[2]-j[1]))
        push!(y_max_tmp, samples_ps.time_integration[i]/(j[4]-j[3]))

        label_1 = "_nolegend_"
        label_2 = "_nolegend_"

    end

    ax.legend(loc="upper left", frameon=false, framealpha=0.8, ncol=1, bbox_to_anchor=(1.02, 1.0),)
    
    ax.set_xlim(x_min, x_max);
    ax.set_ylim(y_min, ceil(maximum(y_max_tmp)));

    ax.set_ylabel("CPU Time", labelpad=10,  size=10)
    ax.set_xlabel("Wall-clock time [s]", labelpad=10,  size=12)
    
end




"""
Plot separated timelines of wall-clock times from different workers. 
"""
function plot_separated_timeline(samples_ps::Union{T,MCMC_Samples}; size=(10, 8)) where {T<:NamedTuple}
    
    minimum_timestamp = minimum(minimum(samples_ps.timestamps))
    timestamps_shifted = [(i .- minimum_timestamp)*10^-9  for i in samples_ps.timestamps]
    maximum_timestamp  = maximum(maximum(timestamps_shifted))
    maximum_cpu_time = maximum([samples_ps.time_mcmc..., samples_ps.time_integration...])
    
    proc_ids = sort(unique(samples_ps.proc_id))
    n_workers = length(proc_ids)
    
    x_min = 0
    x_max = maximum_timestamp
    y_min = 0
    y_max_tmp = Float64[]
    
    label_1 = "Sampling"
    label_2 = "Integration"
    
    fig, ax = plt.subplots(n_workers,1, sharex=true, figsize=size)
    fig.subplots_adjust(hspace=0.0, wspace=0.00)

    for (i,j) in enumerate(timestamps_shifted)
    
        r1 = matplotlib.patches.Rectangle([j[1], 0.0], j[2]-j[1], samples_ps.time_mcmc[i]/(j[2]-j[1]), fill=true, 
            linewidth=0.8, color="royalblue", alpha=0.5, label=label_1) 
        ax[samples_ps.proc_id[i] - 1].add_patch(r1)

        r2 = matplotlib.patches.Rectangle([j[3], 0.0], j[4]-j[3], samples_ps.time_integration[i]/(j[4]-j[3]), 
            fill=true, linewidth=0.8, color="peru", alpha=0.5, label=label_2) 
        ax[samples_ps.proc_id[i] - 1].add_patch(r2)

        push!(y_max_tmp, samples_ps.time_mcmc[i]/(j[2]-j[1]))
        push!(y_max_tmp, samples_ps.time_integration[i]/(j[4]-j[3]))

        label_1 = "_nolegend_"
        label_2 = "_nolegend_"

    end

     ax[1].legend(loc="upper left", frameon=false, framealpha=0.8, ncol=1, bbox_to_anchor=(1.02, 1.0),)

    for i in 1:n_workers
        ax[i].set_xlim(x_min, x_max);
        ax[i].set_ylim(0, ceil(maximum(y_max_tmp)));
        ax[i].set_ylabel("w$i", labelpad=10,  size=10)

    #     ax[i].set_yticks(range(0, stop=floor(maximum(y_max_tmp)), step=1), minor=false)
        ax[i].grid(axis="y", which="both", alpha=0.2, )

        if i<n_workers
            ax[i].get_xaxis().set_visible(false) 
        end
    end

    fig.text(0.06, 0.5, "CPU time / Wall-clock time", va="center", rotation="vertical", size=12, weight="bold")
    ax[n_workers].set_xlabel("Wall-clock time [s]", labelpad=12,  size=12, weight="bold")
    
end

"""
Corner plot of samples with space partitioning (right-top w/o rewriting, left-bottom w/ reweighting). 
"""
function corner_plots(samples::AbstractArray, sample_weights_r::AbstractArray, sample_weights_o::AbstractArray, dim_indices::AbstractArray, dim_names::AbstractArray;
        save_plot=false,
        FILE_NAME = "density_function.pdf",
        N_bins = 100,
        figsize = (11, 11),
        levels_quantiles = [0.4, 0.7, 0.8, 0.9, 0.99, 1,], 
        hist_color = plt.cm.tab10(1), 
        colors = vcat([0 0 0 0.3], plt.cm.YlOrRd(range(0, stop=1, length=10))[2:end,:]),  #vcat([1 1 1 1], plt.cm.Blues(range(0, stop=1, length=20))[2:end,:]),
        kwargs...,
    )
    
    N = length(dim_indices)
    bins=[]
    fig, ax = plt.subplots(N,N, figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for idx in 1:N
        dim_idx = dim_indices[idx]
        bins_tmp = range(minimum(samples[dim_idx,:]), stop=maximum(samples[dim_idx,:]), length=N_bins)
        push!(bins, bins_tmp)
#         ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights_o/sum(sample_weights_o), bins=bins_tmp, color=hist_color, alpha=0.4)
        ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights_r/sum(sample_weights_r), bins=bins_tmp,  color="lightgray", alpha=1, linewidth=0.9) #histtype="step",
        ax[idx, idx].set_xlim(first(bins_tmp),last(bins_tmp))
    end
    
    for i in 2:N, j in 1:(i-1)
        
        dim_x = dim_indices[j]
        dim_y = dim_indices[i]
        
        histogram_2D_r = fit(Histogram, (samples[dim_x,:],samples[dim_y,:]), weights(sample_weights_r), (bins[j], bins[i]))
        histogram_2D_r = normalize(histogram_2D_r, mode=:probability)
        
        histogram_2D_o = fit(Histogram, (samples[dim_y,:],samples[dim_x,:]), weights(sample_weights_o), (bins[i], bins[j]))
        histogram_2D_o = normalize(histogram_2D_o, mode=:probability)
        
#         levels=quantile([histogram_2D_r.weights...], levels_quantiles)
        min_v_1 = minimum(histogram_2D_r.weights[histogram_2D_r.weights .> 0])
        weights_1_filtered = replace( x-> x<10*min_v_1 ? NaN : x, histogram_2D_r.weights')
        
        min_v_2 = minimum(histogram_2D_o.weights[histogram_2D_o.weights .> 0])
        weights_2_filtered = replace( x-> x<10*min_v_2 ? NaN : x, histogram_2D_o.weights')
        
        ax[i,j].pcolormesh(midpoints(histogram_2D_r.edges[1]), midpoints(histogram_2D_r.edges[2]), weights_1_filtered, cmap="RdYlBu_r" , rasterized=true) #bottom | ColorMap(colors) "RdYlBu_r"
        ax[i,j].set_xlim(first(bins[j]),last(bins[j]))
        ax[i,j].set_ylim(first(bins[i]),last(bins[i]))
        
        min_v_2 = minimum(histogram_2D_o.weights[histogram_2D_o.weights .> 0])
        ax[j,i].pcolormesh(midpoints(histogram_2D_o.edges[1]), midpoints(histogram_2D_o.edges[2]), weights_2_filtered,  cmap="RdYlBu_r" , rasterized=true) # top
        
        ax[j,i].set_xlim(first(bins[i]),last(bins[i]))
        ax[j,i].set_ylim(first(bins[j]),last(bins[j]))
        
        
    end
    
    for i in 1:N, j in 1:N
        if 1<i<N 
            ax[i,j].get_xaxis().set_visible(false)
        elseif i==1 
            ax[i,j].xaxis.tick_top()
            ax[i,j].xaxis.set_label_position("top")
            ax[i,j].set_xlabel(dim_names[j])
        else
            ax[i,j].set_xlabel(dim_names[j]) 
        end
        
        if j == i || N>j>1
            # nothing inside 
            ax[i,j].get_yaxis().set_visible(false) 
        elseif j==N
            # right labels
            ax[i,j].set_ylabel(dim_names[i])
            ax[i,j].yaxis.set_label_position("right")
            ax[i,j].yaxis.tick_right()
        else
            #left labels
            ax[i,j].set_ylabel(dim_names[i])
        end
    end
    
    if save_plot 
        fig.savefig(FILE_NAME, bbox_inches = "tight", ) #dpi=500 
    end
        
end


function corner_plots(data::T, dim_indices::AbstractArray, dim_names::AbstractArray; kwargs...) where {T<:NamedTuple}
    smpl_par = collect(hcat(data.samples...))
    w_o = data.weights_o
    w_r =  data.weights_r
    corner_plots(smpl_par, w_r, w_o, dim_indices, dim_names; kwargs...)
end

function corner_plots(data::MCMC_Samples, dim_indices::AbstractArray, dim_names::AbstractArray; kwargs...)
    smpl_par = collect(hcat(data.samples...))
    w_o = data.weights_o
    w_r =  data.weights_r
    corner_plots(smpl_par, w_r, w_o, dim_indices, dim_names; kwargs...)
end
 

function corner_plots(samples::AbstractArray, tree::Node, sample_weights_r::AbstractArray, sample_weights_o::AbstractArray, dim_indices::AbstractArray, dim_names::AbstractArray;
        save_plot=false,
        FILE_NAME = "density_function.pdf",
        N_bins = 100,
        figsize = (11, 11),
        levels_quantiles = [0.4, 0.7, 0.8, 0.9, 0.99, 1,], 
        hist_color = plt.cm.tab10(1), 
#         colors = vcat([0 0 0 0.3], plt.cm.YlOrRd(range(0, stop=1, length=10))[2:end,:]),  #vcat([1 1 1 1], plt.cm.Blues(range(0, stop=1, length=20))[2:end,:]),
#         colors = vcat([1 1 1 1], plt.cm.Blues(range(0, stop=1, length=length(levels_quantiles)))[2:end,:]),
        kwargs...,
    )
    
    N = length(dim_indices)
    bins=[]
    fig, ax = plt.subplots(N,N, figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for idx in 1:N
        dim_idx = dim_indices[idx]
        bins_tmp = range(minimum(samples[dim_idx,:]), stop=maximum(samples[dim_idx,:]), length=N_bins)
        push!(bins, bins_tmp)
#         ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights_o/sum(sample_weights_o), bins=bins_tmp, color=hist_color, alpha=0.4)
        ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights_r/sum(sample_weights_r), bins=bins_tmp,  color="lightgray", alpha=1, linewidth=0.9) #histtype="step",
        ax[idx, idx].set_xlim(first(bins_tmp),last(bins_tmp))
    end
    
    for i in 2:N, j in 1:(i-1)
        
        dim_x = dim_indices[j]
        dim_y = dim_indices[i]
        
        histogram_2D_r = fit(Histogram, (samples[dim_x,:],samples[dim_y,:]), weights(sample_weights_r), (bins[j], bins[i]))
        histogram_2D_r = normalize(histogram_2D_r, mode=:probability)
        
        histogram_2D_o = fit(Histogram, (samples[dim_y,:],samples[dim_x,:]), weights(sample_weights_o), (bins[i], bins[j]))
        histogram_2D_o = normalize(histogram_2D_o, mode=:probability)
        
        min_v_1 = minimum(histogram_2D_r.weights[histogram_2D_r.weights .> 0])
        weights_1_filtered = replace( x-> x<10*min_v_1 ? NaN : x, histogram_2D_r.weights')
        
        min_v_2 = minimum(histogram_2D_o.weights[histogram_2D_o.weights .> 0])
        weights_2_filtered = replace( x-> x<10*min_v_2 ? NaN : x, histogram_2D_o.weights')
        
        ax[i,j].pcolormesh(midpoints(histogram_2D_r.edges[1]), midpoints(histogram_2D_r.edges[2]), weights_1_filtered, cmap="Blues", rasterized=true) #bottom | ColorMap(colors) "RdYlBu_r"
        ax[i,j].set_xlim(first(bins[j]),last(bins[j]))
        ax[i,j].set_ylim(first(bins[i]),last(bins[i]))
        
        min_v_2 = minimum(histogram_2D_o.weights[histogram_2D_o.weights .> 0])
        ax[j,i].pcolormesh(midpoints(histogram_2D_o.edges[1]), midpoints(histogram_2D_o.edges[2]), weights_2_filtered,  cmap="Blues" , rasterized=true) # top
        plot_tree(tree, [dim_y,dim_x], ax[j,i]; kwargs...,)
        
        ax[j,i].set_xlim(first(bins[i]),last(bins[i]))
        ax[j,i].set_ylim(first(bins[j]),last(bins[j]))
        
        
    end
    
    for i in 1:N, j in 1:N
        if 1<i<N 
            ax[i,j].get_xaxis().set_visible(false)
        elseif i==1 
            ax[i,j].xaxis.tick_top()
            ax[i,j].xaxis.set_label_position("top")
            ax[i,j].set_xlabel(dim_names[j])
        else
            ax[i,j].set_xlabel(dim_names[j]) 
        end
        
        if j == i || N>j>1
            # nothing inside 
            ax[i,j].get_yaxis().set_visible(false) 
        elseif j==N
            # right labels
            ax[i,j].set_ylabel(dim_names[i])
            ax[i,j].yaxis.set_label_position("right")
            ax[i,j].yaxis.tick_right()
        else
            #left labels
            ax[i,j].set_ylabel(dim_names[i])
        end
    end
    
    if save_plot 
        fig.savefig(FILE_NAME, bbox_inches = "tight", ) #dpi=500 
    end
        
end

function corner_plots(data::MCMC_Samples, tree::Node, dim_indices::AbstractArray, dim_names::AbstractArray; kwargs...) 
    smpl_par = collect(hcat(data.samples...))
    w_o = data.weights_o
    w_r =  data.weights_r
    corner_plots(smpl_par, tree, w_r, w_o, dim_indices, dim_names; kwargs...)
end

# function corner_plots(data::MCMC_Samples, tree::Node, dim_indices::AbstractArray, dim_names::AbstractArray; kwargs...) 
#     smpl_par = collect(hcat(data.samples...))
#     w_o = data.weights_o
#     w_r =  data.weights_r
#     corner_plots(smpl_par, tree, w_r, w_o, dim_indices, dim_names; kwargs...)
# end

function corner_plots(smpl::T, dim_indices::AbstractArray, dim_names::AbstractArray;
        save_plot=false,
        figsize = (11, 11),
        FILE_NAME = "samples.pdf",
        N_bins = 40,
        levels_quantiles = [0.3, 0.7, 0.8, 0.9, 0.99, 1,], 
        hist_color = plt.cm.Blues(0.7), 
        colors = vcat([1 1 1 1], plt.cm.Blues(range(0, stop=1, length=length(levels_quantiles)))[2:end,:])
    ) where {T<:DensitySampleVector}
    
    samples = flatview(unshaped.(smpl.v))
    sample_weights = smpl.weight
    
    samples_mode = mode(smpl)
    samples_mode = unshaped(samples_mode)
    
    N = length(dim_indices)
    bins=[] 
    fig, ax = plt.subplots(N,N, figsize=figsize)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for idx in 1:N
        dim_idx = dim_indices[idx]
        bins_tmp = range(minimum(samples[dim_idx,:]), stop=maximum(samples[dim_idx,:]), length=N_bins)
        push!(bins, bins_tmp)
        ax[idx, idx].hist(samples[dim_idx,:], weights=sample_weights, bins=bins_tmp, color=hist_color)
        ax[idx, idx].axvline(samples_mode[dim_idx], c="red", label="Truth", alpha=1, lw=1, ls="--")
        ax[idx, idx].set_xlim(first(bins_tmp),last(bins_tmp))
    end
    
    for i in 2:N, j in 1:(i-1)
        dim_x = dim_indices[j]
        dim_y = dim_indices[i]
        histogram_2D = fit(Histogram, (samples[dim_x,:],samples[dim_y,:]), weights(sample_weights), (bins[j], bins[i]))
        histogram_2D = LinearAlgebra.normalize(histogram_2D, mode=:probability)
        
        levels=quantile([histogram_2D.weights...], levels_quantiles)
        
        ax[i,j].contourf(midpoints(histogram_2D.edges[1]), midpoints(histogram_2D.edges[2]), histogram_2D.weights', levels=levels, colors=colors)
        ax[i,j].axvline(samples_mode[dim_x], c="red", label="Truth", alpha=1, lw=1, ls="--")
        ax[i,j].axhline(samples_mode[dim_y], c="red", label="Truth", alpha=1, lw=1, ls="--")
        ax[i,j].set_xlim(first(bins[j]),last(bins[j]))
        ax[i,j].set_ylim(first(bins[i]),last(bins[i]))
        ax[j,i].set_visible(false)
        
    end
    
    for i in 1:N, j in 1:N
        if i < N 
            ax[i,j].get_xaxis().set_visible(false)
        else
            ax[i,j].set_xlabel(dim_names[j])
        end
        
        if j == i || j>1
           ax[i,j].get_yaxis().set_visible(false) 
        else
            ax[i,j].set_ylabel(dim_names[i])
        end
    end
    
    if save_plot 
        fig.savefig(FILE_NAME, bbox_inches = "tight")
    end
        
end