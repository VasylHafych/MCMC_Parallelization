# this code is working, but I do not know if it does what is to do 

using MPIClusterManagers
using Distributed
using KDTree
using CPUTime

manager=MPIManager(np=4)
addprocs(manager)

# import MPI
# MPI.Init()
# comm = MPI.COMM_WORLD

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


    # *** Density Function ***
    true_param =(μ1=2, μ2=-2, σ=0.13)

    function g(x::AbstractArray; true_param=true_param)
        tmp = 1
        for i in eachindex(x)
            if i > 2
                tmp *= pdf(Cauchy(true_param.μ1 + true_param.μ2, true_param.σ), x[i])
            else 
                tmp *= 0.5*pdf(Cauchy(true_param.μ1, true_param.σ), x[i]) + 0.5*pdf(Cauchy(true_param.μ2, true_param.σ), x[i])
            end
        end
        return tmp
    end

    function LogTrueIntegral(N; max = max_v, min=min_v,  true_param=true_param) 
        tmp = 0
        for i in 1:N
            if i > 2
                tmp += log(cdf(Cauchy(true_param.μ1 + true_param.μ2,true_param.σ), max_v) - cdf(Cauchy(true_param.μ1 + true_param.μ2,true_param.σ), min_v))
            else 
                tmp += log(cdf(Cauchy(true_param.μ1,true_param.σ), max_v) - cdf(Cauchy(true_param.μ1 ,true_param.σ), min_v))
            end
        end
        return tmp
    end

    N = 6
    min_v = -3.
    max_v = 3.
    lgV = N*log(max_v-min_v); 
    likelihood = params -> LogDVal((log(g(params.a))))

    # *** Density Function ***

    BATPar.make_named_prior(i) = BAT.NamedTupleDist( a =  [[i[j,1]..i[j,2] for j in 1:size(i)[1]]...])

end
    

prior = NamedTupleDist(a = [[min_v .. max_v for i in 1:N]...],);
posterior = PosteriorDensity(likelihood, prior);

nnsamples = 450
nnchains = 75

samples, stats = bat_sample(posterior, (nnsamples, nnchains), MetropolisHastings(),);

smpl = flatview(unshaped.(samples.v))
weights_LogLik = samples.logd
weights_Histogram = samples.weight;

data_kdtree = Data(smpl[:,1:5:end], weights_Histogram[1:5:end], weights_LogLik[1:5:end]);

KDTree.evaluate_total_cost(data::Data) = KDTree.cost_f_1(data)

output, cost_array = DefineKDTree(data_kdtree, [1,2,3,4], 20);

extend_tree_bounds!(output, repeat([min_v], N), repeat([max_v], N))

bounds_part = extract_par_bounds(output)

nnsamples = 10^4
nnchains = 10

tuning = AdaptiveMetropolisTuning(
    λ = 0.5,
    α = 0.15..0.35,
    β = 1.5,
    c = 1e-4..1e2
)

burnin = MCMCBurninStrategy(
    max_nsamples_per_cycle = 4000,
    max_nsteps_per_cycle = 4000,
    max_time_per_cycle = 25,
    max_ncycles = 200
)

algorithm = MetropolisHastings();

samples_parallel = bat_sample_parallel(likelihood, bounds_part, (nnsamples, nnchains), algorithm, tuning=tuning, burnin=burnin);

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

@save "/Users/vhafych/MPP-Server/gitrepos/MCMC_Parallelization/Generated_Data/samples_ps.jld" samples_ps;


# MPI.Barrier(comm)

exit()