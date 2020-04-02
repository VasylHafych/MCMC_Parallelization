using ValueShapes
using ArraysOfArrays
using StatsBase 
using LinearAlgebra
using Statistics
using BAT
using Distributions 
using IntervalSets

using HCubature
using JLD
using CPUTime

using BATPar
using KDTree

# ***************************************
# Distribtions: 
# ***************************************

PATH = "Generated_Data/normal_1.jld" # path where data will be saved

f(x;) = prod(pdf.(Normal(), x))
LogTrueIntegral(N) = 0.0
N = 7
min_v = -30.
max_v = 30.
lgV = N*log(max_v-min_v); 
true_int = LogTrueIntegral(N)

# ***************************************

algorithm = MetropolisHastings()
log_likelihood = params -> LogDVal((log(f(params.a))))

nchains = 3
nsamples = 10^4
max_time = 100
max_nsteps = 10 * nsamples
prior_bounds = [min_v, max_v] 

BATPar.make_named_prior(i) = NamedTupleDist( a =  [[i[j,1]..i[j,2] for j in 1:size(i)[1]]...])
KDTree.evaluate_total_cost(data::Data) = KDTree.cost_f_1(data)

function run_integrations(cut_range::StepRange{Int64,Int64}, n_repeat::Int64; 
    nchains = nchains,
    N=N, 
    true_int=true_int,
    nsamples = nsamples,
    max_time = max_time,
    max_nsteps = max_nsteps,
    log_likelihood = log_likelihood,
    prior_bounds = prior_bounds)
	 
    # information that we want to track 
	integrals_ahmi_array = Vector{Float64}()
	integrals_true_array = Vector{Float64}()
	cut_array = Vector{Int64}()
	uns_ahmi_array = Vector{Float64}()
	mcmc_time_array = Vector{Float64}()
	n_samples_array = Vector{Int64}()
    
    prior = NamedTupleDist(a = [[prior_bounds[1] .. prior_bounds[2] for i in 1:N]...],)
	posterior = PosteriorDensity(log_likelihood, prior)
    

	for cut_run in cut_range
        
		for n_run in 1:n_repeat
            
            samples, stats = bat_sample(posterior, (10^2, 20), MetropolisHastings());
    
            smpl = flatview(unshaped.(samples.v))
            weights_LogLik = samples.logd
            weights_Histogram = samples.weight;
            data_kdtree = Data(smpl[:,1:end], weights_Histogram[1:end], weights_LogLik[1:end]);

            output, cost_array = DefineKDTree(data_kdtree, [1,2,3,], cut_run);
            extend_tree_bounds!(output, repeat([prior_bounds[1]], N), repeat([prior_bounds[2]], N))

			@show cut_run, n_run
            
            bounds_part = extract_par_bounds(output)

            algorithm = MetropolisHastings();

			mcmc_ex_time = @CPUelapsed begin 
                samples_parallel = bat_sample_parallel(log_likelihood, bounds_part, (nsamples, nchains), algorithm);
            end

			ahmi_integral_run =[sum(samples_parallel.weights_r), sqrt(sum((samples_parallel.uncertainty).^2))]
            
            push!(cut_array, cut_run)
			push!(n_samples_array, length(samples_parallel.weights_o))
			push!(mcmc_time_array, mcmc_ex_time)
			push!(integrals_ahmi_array, ahmi_integral_run[1])
			push!(integrals_true_array, true_int)
			push!(uns_ahmi_array, ahmi_integral_run[2])

		end
		
		# Save all data after each dimension. This protects from losing data if AHMI/MCMC fails.
		
		save_data(deepcopy(n_samples_array), 
            deepcopy(integrals_true_array), 
            deepcopy(integrals_ahmi_array), 
            deepcopy(uns_ahmi_array), 
            deepcopy(mcmc_time_array), 
            deepcopy(cut_array), 
            deepcopy(cut_run)
        )
		
    end
end

function save_data(n_samples_array::Vector{Int64}, 
        integrals_true_array::Vector{Float64}, 
        integrals_ahmi_array::Vector{Float64}, 
        uns_ahmi_array::Vector{Float64}, 
        mcmc_time_array::Vector{Float64},  
        dim_array::Vector{Int64}, 
        dim_run::Int64, 
        PATH=PATH)
    
	x_dms = Int64(length(integrals_ahmi_array)/length(unique(dim_array)))
    y_dms = Int64(dim_array[end]-dim_array[1]+1)
    
    n_samples_array = reshape(n_samples_array, x_dms, y_dms)
	integrals_true_array = reshape(integrals_true_array, x_dms, y_dms)
	integrals_ahmi_array = reshape(integrals_ahmi_array, x_dms, y_dms)
		
	uns_ahmi_array = reshape(uns_ahmi_array, x_dms, y_dms)
	mcmc_time_array = reshape(mcmc_time_array, x_dms, y_dms)
	
    dim_array = reshape(dim_array, x_dms, y_dms)
		
	integrals_ahmi_array = convert(Array{Float64,2}, integrals_ahmi_array)
	uns_ahmi_array = convert(Array{Float64,2}, uns_ahmi_array);
		
	isfile(PATH) && rm(PATH)
	@show "saving"
	save(PATH, 
		"sample_size", n_samples_array, 
		"integrals_ahmi_array", integrals_ahmi_array, 
		"integrals_true_array", integrals_true_array,  
		"uns_ahmi_array", uns_ahmi_array,  
		"dim_array", dim_array, 
		"mcmc_time_array", mcmc_time_array)
end

cut_range = range(0, step=1, stop=30)

@CPUtime run_integrations(cut_range, 20)