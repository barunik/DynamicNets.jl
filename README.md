# DynamicNets.jl


The code has been developed in Julia 1.4.0 version, as a code accompanying the Barunik and Ellington (2020) papers, and provides an estimation and inference for *dynamic networks* measures introduced in the following papers

Baruník, J. and Ellington, M. (2020): *Dynamic Networks in Large Financial and Economic Systems*, manuscript [available here for download](https://ideas.repec.org/p/arx/papers/2007.07842.html) (July 2020)

Baruník, J. and Ellington, M. (2020): *Dynamic Network Risk*, manuscript [available here for download](https://ideas.repec.org/p/arx/papers/2006.04639.html) (Jun 2020)


## Software requirements

[Julia](http://julialang.org/) together with few packages needs to be installed

````julia
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Statistics")
Pkg.add("LinearAlgebra")
Pkg.add("Distributions")
Pkg.add("Plots")
````

# Example of usage
## Time Dynamics of Network Connectedness

This example illustrates how to obtain various dynamic network measures on an example data consising of 4 variable system

NOTE that computation time is growing with number of variables in the system and simulations used to obtain measures. This readme file also includes usage of parallel computing that can help to increase the speed of computations tremendously.

Note the full example is available as an interactive [IJulia](https://github.com/JuliaLang/IJulia.jl) notebook [here](https://github.com/barunik/DistributionalForecasts.jl/blob/master/Example.ipynb)


Load required packages


```julia
using DataFrames, CSV, Statistics, LinearAlgebra, Distributions
using Plots 

# load main functions
include("DynamicNets_functions.jl");
```

Load example data


```julia
data = CSV.read("exampledata.txt",DataFrame,header=false)
data = convert(Matrix{Float64}, data[:,1:4]);
data = sqrt.(data);
```

Function DynNet_time estimates time varying total network connectedness as well as directional connectedness with following inputs and outputs, timing for a 9 variable system and 100 simulations is for MacBook Pro 2020 with 2,3 GHz Quad-Core Intel Core i7 with 32GB 3733 MHz Memory. Note julia compiles functions at the first run so the second run times will be much faster

````julia
C,CI = DynNet_time(data,L,H,W,Nsim,corr)

# INPUTS: L = 2,       number of lags in VAR
#          H=10,       horizons in the FEVD
#          W = 8,       width of kernel bandwidth
#          Nsim = 100,  number of simulations 
#          corr = true, true = diagonal covariance matrix of errors
#                       false = estimated covariance matrix of errors
#
# OUTPUTS:  C           median estimates of Network Connectedness
#                       C has [ ( 1+5xN ) x T ] dimension with N number of variables in the system
#           CI          standard deviation of all C measures
#                       CI has [ ( 1+5xN ) x T ] dimension with N number of variables in the system
# OUTPUT ROWs:
#          1 is Total network connectedness
#          2+0xN...1xN+1 is net TO network connectedness for all N variables
#          2+1xN...2xN+1 is net FROM network connectedness for all N variables
#          2+2xN...3xN+1 is net FROM network connectedness normalised by own shock for all N variables
#          2+3xN...4xN+1 is net TO network connectedness normalised by own shock for all N variables
#          2+4xN...5xN+1 is net directional (difference) network connectedness for all N variables
````


```julia
@time C,CI = DynNet_time(data,2,10,8,100,false);
```

     46.210951 seconds (128.11 M allocations: 72.418 GiB, 5.63% gc time)


Dimension of the estimated measures is always [ ( 1+5xN ) x T ], for N=4 variables and T=1832 time periods, it is


```julia
size(C)
```




    (21, 1832)



The first row of C and CI is holding total dynamic network connectedness meadian and stadard deviation


```julia
plot(C[1,:],ribbon=(1.96*CI[1,:],1.96*CI[1,:]),fillalpha=0.2,color=["black" "grey" "grey"],legend=false,ylim=[0,100],framestyle = :box,
    size=(800,480),title="TVP Network Connectedness")
```




![svg](/readme_files/output_11_0.svg)




```julia
i=3
plot(plot(C[i+1,:],ribbon=(1.96*CI[i+1,:],1.96*CI[i+1,:]),fillalpha=0.2,color=["black" "grey" "grey"],legend=false,ylim=[0,10],framestyle = :box,
    size=(400,400),title="Directional TO for Variable 3"),
    plot(C[2+9+i-1,:],ribbon=(1.96*CI[2+9+i-1,:],1.96*CI[2+9+i-1,:]),fillalpha=0.2,color=["black" "grey" "grey"],legend=false,ylim=[0,10],framestyle = :box,
    size=(400,400),title="Directional FROM for Variable 3"),
size=(1000,400))
```




![svg](/readme_files/output_12_0.svg)




```julia
plot(C[2+4*4+i-1,:],ribbon=(1.96*CI[2+4*4+i-1,:],1.96*CI[2+4*4+i-1,:]),fillalpha=0.2,color=["black" "grey" "grey"],legend=false,ylim=[-4,4],framestyle = :box,
    size=(500,400),title="Directional NET for variable 3")
```




![svg](/readme_files/output_13_0.svg)



## Time Horizon Dynamics

Note that current version of the codes works with 3 possible horizons of user's choice

Function DynNet estimates time varying total network connectedness as well as directional connectedness with following inputs and outputs, timing for a 9 variable system and 33 simulations is for MacBook Pro 2016 with 2.9 GHz Dual-Core Intel Core i5

````julia
C,CI = DynNet(data,horizon1,horizon2,L,W,Nsim,corr)

# INPUTS:  horizon1 = 5  horizon cutting short and medium run
#          horizon2 = 20 horizon cutting medium and long run
#          L = 2,       number of lags in VAR
#          W = 8,       width of kernel bandwidth
#          Nsim = 100,  number of simulations 
#          corr = true, true = diagonal covariance matrix of errors
#                       false = estimated covariance matrix of errors
#
# OUTPUTS:  C           median estimates of Network Connectedness
#                       C has [ (7 + (2*4*2+4)*N ) x T ] dimension with N number of variables in the system
#           CI          standard deviation of all C measures
#                       CI has [ ( 7 + (2*4*2+4)*N ) x T ] dimension with N number of variables in the system
# OUTPUT ROWs:
#          1               is total connectedness (as sum of long+medium+short)
#          2               is long-term connectedness
#          3               is med-term connectedness
#          4               is short-term connectedness
#          5               is within long-term connectedness
#          6               is within med-term connectedness
#          7               is within short-term connectedness
#          7+1+0xN...7+1xN is long-term to connectedness
#          7+1+1xN...7+2xN is med-term to connectedness
#          7+1+2xN...7+3xN is short-term to connectedness
#          7+1+3xN...7+4xN is total net to connectedness
#          7+1+4xN...7+5xN is long-term from connectedness
#          7+1+5xN...7+6xN is med-term from connectedness
#          7+1+6xN...7+7xN is short-term from connectedness
#          7+1+7xN...7+8xN is total net from connectedness
#          7+1+8xN...7+9xN is long-term to connectedness normalised by own shock
#          7+1+9xN...7+10xN is med-term to connectedness normalised by own shock
#          7+1+10xN...7+11xN is short-term to connectedness normalised by own shock
#          7+1+11xN...7+12xN is total net to connectedness normalised by own shock
#          7+1+12xN...7+13xN is long-term from connectedness normalised by own shock
#          7+1+13xN...7+14xN is med-term from connectedness normalised by own shock
#          7+1+14xN...7+15xN is short-term from connectedness normalised by own shock
#          7+1+15xN...7+16xN is total net from connectedness normalised by own shock
#          7+1+16xN...7+17xN is long-term net directional connectedness
#          7+1+17xN...7+18xN is med-term net directional connectedness
#          7+1+18xN...7+19xN is short-term net directional connectedness
#          7+1+19xN...7+20xN is total net directional freq connectedness
````

Example of dynamic horizon specific network with horizons defined as
* short run: 1 - 5 days (up to one week)
* medium run: 5 - 20 days (week up to month)
* long run: 20 + days (more than month)


```julia
@time Chorizon,CIhorizon = DynNet(data,5,20,2,8,33,false);
```

    260.019783 seconds (1.52 G allocations: 174.821 GiB, 4.01% gc time)


Dimension of the estimated measures is always [ (7 + (2x4x2+4)xN ) x T ], for N=9 variables and T=1832 time periods, it is


```julia
size(Chorizon)
```




    (87, 1832)



The first row of C and CI is holding total dynamic network connectedness meadian and stadard deviation
The rows 2,3,4 hold long-term, smedium-term and short-term network connectedness and they always sum to row 1 total


```julia
plot(Chorizon[1,:],ribbon=(1.96*CIhorizon[1,:],1.96*CIhorizon[1,:]),fillalpha=0.2,color=["black" "grey" "grey"],legend=false,ylim=[0,100],framestyle = :box,
    size=(800,480),title="TVP Network Connectedness")
plot!(Chorizon[2,:],ribbon=(1.96*CIhorizon[2,:],1.96*CIhorizon[2,:]),fillalpha=0.2,color=["red" "red" "red"])
plot!(Chorizon[3,:],ribbon=(1.96*CIhorizon[3,:],1.96*CIhorizon[3,:]),fillalpha=0.2,color=["blue" "blue" "blue"])
plot!(Chorizon[4,:],ribbon=(1.96*CIhorizon[4,:],1.96*CIhorizon[4,:]),fillalpha=0.2,color=["green" "green" "green"])
```




![svg](/readme_files/output_22_0.svg)



All other horizon specific directional network measures can be obtained analogously chosing the correct row from the output

# Fast multiple cores computation usage

Computations are costly for large systems with large number of variables and simulations consider the use of multiple cores using following package

````julia
Pkg.add("Distributed")
````

Note that in the paper we estimate system of 496 stocks that required several days of multiple server time, hence it is useful to adapt function to the needs of a specific case. For example you can consider saving the results to files as in following example which distributes the core of  the function

````julia
C,CI = DynNet(data,horizon1,horizon2,L,H,W,Nsim,corr)
````

to multiple cores. The computation can be distributed to multiple cores simply, here is an example running on 48 core server


```julia
using Distributed
addprocs(48);
```


```julia
@everywhere using DataFrames, CSV, Statistics, LinearAlgebra, Distributions

# load main functions
@everywhere include("DynamicNets_functions.jl");

@everywhere data = CSV.read("exampledata.txt",header=false)
@everywhere data = convert(Matrix{Float64}, data[:,1:9]);
@everywhere data = log.(sqrt.(data));
```


```julia
@everywhere Nsim = 96

@everywhere shrinkage = 0.05
@everywhere L = 2

@everywhere T, N = size(data)
@everywhere K = N * L + 1

@everywhere X = zeros(Float64, T - L, K-1)
@everywhere for i in 1:L
	temp = lag0(data, i)
    X[:, (1 + N*(i-1) : i*N)] = temp[(1+L:T), :]
end
@everywhere y = data[(1+L):T, :]
@everywhere T = T - L
@everywhere X = [ones(Float64, T, 1) X]

@everywhere K = N * L + 1

@everywhere SI, PI, a, RI = MinnNWprior(data, T, N, L, shrinkage)
@everywhere weights1 = convert.(Float64, normker(T, 8))
@everywhere priorprec0 = convert.(Float64, inv(PI)) 

for it in 1:T

	out = hcat(pmap(i -> f_all(it, i,5,20, T, N, L, weights1, priorprec0, X, y, SI, PI, a, RI), 1:Nsim)...);

	xmed = median(out,dims=2)
	xmean = mean(out,dims=2)
	xsd = std(out,dims=2)

	CSV.write("results_time_$it.csv", DataFrame([xmed xmean xsd]))

end
```
