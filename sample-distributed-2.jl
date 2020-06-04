# Using MPIClusterManagers.jl

using Distributed, ClusterManagers

ENV["JULIA_NUM_THREADS"] = ENV["SLURM_CPUS_PER_TASK"]
slurm_ntasks = parse(Int, ENV["SLURM_NTASKS"])
addprocs(SlurmManager(slurm_ntasks))

using KDTree
using CPUTime

@everywhere begin 
    using Distributed
    using Distributions 
    using IntervalSets
    using ValueShapes
    using ArraysOfArrays
    using StatsBase 
    using LinearAlgebra
    using BATPar
    using BAT
    using JLD2
end

# @everywhere begin
#     """
#     Caushy Density
#     """
#     true_param =(μ1=4, μ2=-4, σ=0.13)
#     function g(x::AbstractArray; true_param=true_param)
#         tmp = 1
#         for i in eachindex(x)
#             if i > 2
#                 tmp *= pdf(Cauchy(true_param.μ1 + true_param.μ2, true_param.σ), x[i])
#             else 
#                 tmp *= 0.5*pdf(Cauchy(true_param.μ1, true_param.σ), x[i]) + 0.5*pdf(Cauchy(true_param.μ2, true_param.σ), x[i])
#             end
#         end
#         return tmp
#     end
#     function LogTrueIntegral(N; max = max_v, min=min_v,  true_param=true_param) 
#         tmp = 0
#         for i in 1:N
#             if i > 2
#                 tmp += log(cdf(Cauchy(true_param.μ1 + true_param.μ2,true_param.σ), max_v) - cdf(Cauchy(true_param.μ1 + true_param.μ2,true_param.σ), min_v))
#             else 
#                 tmp += log(cdf(Cauchy(true_param.μ1,true_param.σ), max_v) - cdf(Cauchy(true_param.μ1 ,true_param.σ), min_v))
#             end
#         end
#         return tmp
#     end

#     N = 6
#     min_v = -10.
#     max_v = 10.
#     lgV = N*log(max_v-min_v); 
#     likelihood = params -> LogDVal((log(g(params.a))))  
# end

@everywhere begin
    JLD2.@load "MixtureModels/mixture-1.jld" means cov_m n_clusters
    model = MixtureModel(MvNormal[MvNormal(means[i,:], Matrix(Hermitian(cov_m[i,:,:])) ) for i in 1:n_clusters])
    N = 6
    min_v = -100.
    max_v = 100.
    lgV = N*log(max_v-min_v); 
    likelihood = let model=model ;begin params -> LogDVal(logpdf(model, params.a)) end end
end

try
    # Exploration Samples: 
    
    prior_exploration = NamedTupleDist(a = [[min_v .. max_v for i in 1:N]...],);
    posterior_exploration = PosteriorDensity(likelihood, prior_exploration);

    exp_samples = 70
    exp_chains = 30

    exploration_samples = bat_sample(posterior_exploration, (exp_samples, exp_chains), MetropolisHastings(),).result

    # KD Tree:
    data_kdtree = Data(collect(flatview(unshaped.(exploration_samples.v))), exploration_samples.weight, exploration_samples.logd)

    n_partitions = 100
    dims_partition = collect(1:N)

    KDTree.evaluate_total_cost(data::Data) = KDTree.cost_f_1(data)

    partition_tree, _ = DefineKDTree(data_kdtree, dims_partition, n_partitions)

    extend_tree_bounds!(partition_tree, repeat([min_v], N), repeat([max_v], N))

    subspace_boundaries = extract_par_bounds(partition_tree)

    @everywhere BATPar.make_named_prior(i) = BAT.NamedTupleDist( a =  [[i[j,1]..i[j,2] for j in 1:size(i)[1]]...])

    nsamples_per_subspace = 2*10^5
    nchains_per_subspace = 6

    tuning = AdaptiveMetropolisTuning(
        λ = 0.5,
        α = 0.1..0.15,
        β = 1.5,
        c = 1e-4..1e2
    )

    burnin = MCMCBurninStrategy(
        max_nsamples_per_cycle = 5000,
        max_nsteps_per_cycle = 5000,
        max_time_per_cycle = 25,
        max_ncycles = 40
    )

     AHMI_settings = BAT.HMISettings(BAT.cholesky_partial_whitening!,
        30000, 3.0, 0.1, true, 30, true,  Dict("cov. weighted result" => BAT.hm_combineresults_covweighted!));

    algorithm = MetropolisHastings();

    @time samples_parallel = bat_sample_parallel(likelihood, subspace_boundaries, (nsamples_per_subspace, nchains_per_subspace), 
        algorithm, tuning=tuning, burnin=burnin, settings=AHMI_settings, )

    samples_ps = (samples = samples_parallel.samples,
                weights_o = samples_parallel.weights_o,
                weights_r = samples_parallel.weights_r,
                log_lik = samples_parallel.log_lik,
                space_ind = samples_parallel.space_ind,
                uncertainty = samples_parallel.uncertainty,
                integrals = samples_parallel.integrals,
                time_mcmc = samples_parallel.time_mcmc,
                time_integration = samples_parallel.time_integration,
                proc_id = samples_parallel.proc_id,
                n_threads = samples_parallel.n_threads,
                timestamps = samples_parallel.timestamps)

    @save "Generated_Data/test-3.jld" samples_ps;

finally
   rmprocs.(workers())
end

@info "Done."