
#**************************************************************************
# by Michael Ellington, Lubos Hanus and Jozef Barunik
#**************************************************************************

## Set of functions to estimate TVP QBLL

function MinnNWprior(Y, T, N, L, shrinkage)
  K = N * L + 1
  SI = vcat(zeros(1, N), 0.1 .* diagm(0 => ones(N)), zeros((L-1)*N, N))
  PI = zeros(K, 1)
  sigma_sq = zeros(N, 1)
  
  for i in 1:N
      # Create lags of dependent variable
      Y_i = mlag2(Y[:, i], L)
      Y_i = Y_i[(L+1):T, :]
      X_i = [ones(T - L, 1) Y_i]
      y_i = Y[(L+1):T, i]
      # OLS estimates of i-th equation
      alpha_i = (X_i' * X_i) \ (X_i' * y_i)
      sigma_sq[i, 1] = (1.0 ./ (T-L+1)) * (y_i .- X_i * alpha_i)' * (y_i .- X_i * alpha_i)
  end

  s = 1.0 ./ sigma_sq

  for ii=1:L
    PI[(2 + N * (ii-1)):(1+N*ii)] = (shrinkage * shrinkage) * s / (ii * ii)
  end

  PI[1] = 10^2 # prior variance for constant is loose
  PI2 = Diagonal(vec(PI)) .* ones(K, K)
 
  # now for Wishart priors following Petrova (2018)
  a = max(N+2, N+2*8-T)
  RI = (a-N-1) * sigma_sq
  RI2 = Diagonal(vec(RI)) .* ones(N, N)

  return(SI, PI2, a, RI2)
end

# MinnNWprior(randn(100, 2), 100, 2, 3, 0.05)

function normker(T, H)
  ww = zeros(T, T)
  for j in 1:T
    for i in 1:T
      z = (i - j)/H
      ww[i, j] = (1.0 / sqrt(2.0 * pi)) * exp((-1.0 / 2.0) * (z * z))
    end     
  end 

  s = sum(ww, dims=2)
  adjw = zeros(T, T)  

  for k in 1:T
    adjw[k, :] = ww[k, :] / s[k]
  end

  cons = sum(adjw .^ 2.0, dims = 2)

  for k in 1:T
    adjw[k, :] = (1.0 / cons[k]) * (adjw[k, :])
  end

  return adjw
end


function mlag2(X, p)
  if ndims(X) == 1
    Traw = length(X)
    N = 1
  else 
    Traw, N = size(X)
  end

  Xlag = zeros(Traw, N*p)
  for ii in 1:p
      Xlag[(p+1):Traw, (N*(ii - 1)+1):(N*ii)] = X[(p+1-ii):(Traw-ii), 1:N]
  end
  return Xlag
end


function lag0(x, p)
  R, C = size(x)
  # Take the first R-p rows of matrix x
  x1 = x[1:(R-p), :]
  # Preceed them with p rows of zeros and return
  return vcat(zeros(p, C), x1)
end

function Lsize(N, nsd)
  LL = (nsd/N  - 1)/N
  convert(Int, LL)
end

# GET CONNECTEDNESS FUNCTIONS

function varcompanion(A, ndet, n, p)
	# create companion matrix of A
	A = A[:, (ndet+1):end]
	A = [A; [Diagonal(ones(n*(p-1))) zeros(n*(p-1),n)]]
	return A
end

function getCoefsStable(bayesalpha, bayesgamma, bayessv, BB, N, L)

	B0 = zeros(N, N*L+1)
	A0 = zeros(N, N)

	Ficom = zeros(N*L, N*L)

    mm = 0
    while mm < 1
		A0 = rand(InverseWishart(bayesalpha, bayesgamma))
		nu = randn(N*L+1, N)

		B0 = (BB + cholesky(Symmetric(bayessv).*1.0).U' * (nu * (cholesky(A0).U)))'
		Ficom[1:N, :] = B0[:, 2:(N*L+1)]
		for pp in 2:L
			Ficom[(1 + N * (pp-1)):(pp*N), (1 + N * (pp-2)):(N*(pp-1))] = diagm(0 => ones(N)); # companion without constant 
		end
		maxEig = maximum(abs.(eigvals(Ficom)))
		if maxEig < .999 # check stability of draw
			stabInd = 1
			mm = 1
		end
    end
    return B0, A0
end

function get_GIRF(B, A0, ND::Int, N::Int, L::Int, HORZ::Int,corr)
	# Get GIRFs as in Equation (10) KPP(1996)
	# B is N, N*L+1 coefficient matrix
	# A0 is N x N covariance matrix 
	# ND = 1 a constant or not. # L is lag length

	if corr == true
		A0 = Diagonal(A0);
	end 

	B = convert(Array{Float64}, varcompanion(B, ND, N, L))
	J = convert(Array{Float64}, [Diagonal(ones(N)) zeros(N, N*(L-1))])

	ir1 = Array{Float64}(undef, N, N, HORZ+1);
	wold = Array{Float64}(undef, N, N, HORZ+1);

	# GET MA coefficients
	jT = J'
	bh = B ^ 0

	@views @inbounds for h in 0:HORZ
		wold[:, :, h+1] = J * ((bh) * jT)
		bh = bh * B
	end

	A0 = (1.0 ./ sqrt.(diag(A0))) .* A0

	@views @inbounds for h in 1:(HORZ+1)
		@inbounds for i in 1:N
			ir1[:, i, h] = A0[i, :]' * wold[:, :, h]';
		end
	end


	return ir1, wold
end

function get_GIRF_fast(B, A0, ND::Int, N::Int, L::Int, HORZ::Int)
	# Get WOLD .. 
	# B is N, N*L+1 coefficient matrix
	# A0 is N x N covariance matrix 
	# ND = 1 a constant or not. # L is lag length

	B = convert(Array{Float64}, varcompanion(B, ND, N, L))
	J = convert(Array{Float64}, [Diagonal(ones(N)) zeros(N, N*(L-1))])

	wold = Array{Float64}(undef, N, N, HORZ+1);

	# GET MA coefficients
	jT = J'

	bh = B ^ 0

	@views @inbounds for h in 0:HORZ
		wold[:, :, h+1] = J * (bh * jT)
		bh = bh * B
	end

	return wold
end

function var_decomp(nvars, nsteps, ir)
	# calculates variance decomposition and accumulate vardeco
	
	resp6 = Array{Float64}(undef, nvars, nvars, nsteps);
	resp7 = zeros(nsteps, 1);
	vardecomp = Array{Float64}(undef, nvars, nvars, nsteps);

	@inbounds for j in 1:nvars
		@inbounds for i in 1:nvars
			# variance of the forecast error: conditional variance    
			resp6[i, j, :] = @views cumsum((ir[i, j, :] .* ir[i, j, :]));
		end
	end

	@inbounds for j in 1:nvars
		@inbounds for i in 1:nvars
			@inbounds for k in 1:nsteps
				resp7[k, 1] = @views sum(resp6[i, :, k]);
			end
			# conditional/unconditional variance
			vardecomp[i, j, :] = @views (resp6[i, j, :])' ./ resp7'; 
		end
	end

	return(vardecomp)
end

function get_timenet(N, HO, irf)

	fev = var_decomp(N, HO, irf);
	fev = fev[:, :, end];
	FF = sum(fev);
	timecon = 100.0 .* (1.0 .- tr(fev) ./ FF);

	trT = zeros(N)
	ttT = zeros(N)
	DCRTnorm = zeros(N) 
	DCTTnorm = zeros(N) 

	for i in 1:N

		trT[i] = 100.0 .* ( sum(fev[i, :]) .- fev[i, i] ) ./FF
		ttT[i] = 100.0 .* ( sum(fev[:, i]) .- fev[i, i] ) ./FF
        
		DCRTnorm[i]= sum(fev[i, :] ./ fev[i, i])
		DCTTnorm[i]= sum(fev[:, i] ./ fev[i, i])

	end

	NDCT = ttT .- trT

	return (timecon, trT, ttT, DCRTnorm, DCTTnorm, NDCT)
end


function oneTimePrior(kk, weights1, priorprec0, X, y, SI, PI, a, RI)

	w = weights1[kk, :]
	bayesprec = (priorprec0 .+ X' * Diagonal(w) * X)
	bayessv = inv(bayesprec)
	BB = bayessv * ((X' * Diagonal(w)) * y .+ priorprec0 * SI)
	
	bayesalpha = a + sum(w)
	g1 = SI' * priorprec0 * SI
	g2 = y' * Diagonal(w) * y
	g3 = BB' * bayesprec * BB

	bayesgamma = RI + g1 + g2 - g3
	bayesgamma = 0.5 * bayesgamma + 0.5 * bayesgamma'  # it is symmetric but just in case
	return bayesalpha, bayessv, bayesgamma, BB
end


function get_dynnet(wo, TT, sig,corr,cut1::Int,cut2::Int)

	Tw = 200 # Define frequency window;
	omeg = LinRange(0.0, pi, Tw) # create equally spaced line from 0 to pi in 261 intervals

	# Define bands
	omeg2 = pi ./ omeg; 
	d1 = omeg2 .> cut2       # long term equals (20,260+] days
	d2 = (omeg2 .<= cut2) .* (omeg2 .> cut1)  # medium term equals (5,20] days
	d3 = omeg2 .<= cut1    # short term equals [1,5] days

	N = size(wo, 1)
	HO = size(wo, 3)

	if corr == true
		diag_sig = Diagonal(sig);
	end 
	
	if corr == false
		diag_sig = sig;
	end 
	
	expnnom = exp.(-im .* repeat(omeg, 1, HO) .* repeat((1:HO)', Tw, 1));
	expnnom = convert.(Complex{Float32}, expnnom)

	Omeg2 = Array{Float64}(undef, N, N)
	@views @inbounds for hh in 1:HO
		Omeg2 += wo[:,:,hh] * (diag_sig * wo[:,:,hh]')
	end
	Omeg2 = diag(Omeg2)

	wo = convert.(Float32, wo);
	FC = Array{Float64}(undef, N, N, Tw);

	GI = Array{Complex{Float32}}(undef, N, N);

	@views @inbounds for w in 1:Tw
		
		fill!(GI, 0.0);
		for nn in 1:HO
			GI .+= wo[:, :, nn] .* expnnom[w, nn];
		end

		PS = abs2.(GI * diag_sig); 

		@inbounds for k in 1:N 
			@inbounds for j in 1:N 
				FC[j, k, w] = PS[j, k] ./ (Omeg2[j] .* sig[k, k]);
			end
		end
	end

	PP1 = dropdims(sum(FC, dims = 3), dims=3);
	
	@views for w in 1:Tw
		for j in 1:N
			FC[j, :, w] = FC[j, :, w] ./ sum(PP1[j, :]); 
		end
	end

	thetainf = dropdims(sum(FC, dims = 3), dims=3);

	### BANDS : d1 d2 d3, Long, Medium, Short
	# theta_{d_i} summed over bands
	temp1 = dropdims(sum(FC[:, :, d1], dims = 3), dims=3); 
	temp2 = dropdims(sum(FC[:, :, d2], dims = 3), dims=3);
	temp3 = dropdims(sum(FC[:, :, d3], dims = 3), dims=3);

	for j in 1:N
		sumthetaj = sum(thetainf[j, :]);
		temp1[j, :] = temp1[j, :] ./ sumthetaj;
		temp2[j, :] = temp2[j, :] ./ sumthetaj;
		temp3[j, :] = temp3[j, :] ./ sumthetaj;
	end

	## GET Net Directional Connectedness and Transmitters + Recievers
	trL = zeros(N)
	ttL = zeros(N)
	trM = zeros(N)
	ttM = zeros(N)
	trS = zeros(N)
	ttS = zeros(N)
	trT = zeros(N)
	ttT = zeros(N)
	DCRL = zeros(N)
	DCTL = zeros(N) 
	DCRM = zeros(N) 
	DCTM = zeros(N) 
	DCRS = zeros(N) 
	DCTS = zeros(N) 
	DCRT = zeros(N) 
	DCTT = zeros(N)
	DCRLnorm = zeros(N)
	DCTLnorm = zeros(N) 
	DCRMnorm = zeros(N) 
	DCTMnorm = zeros(N) 
	DCRSnorm = zeros(N) 
	DCTSnorm = zeros(N) 
	DCRTnorm = zeros(N) 
	DCTTnorm = zeros(N) 
	
	for i in 1:N
		trL[i] = sum(temp1[i, :]) .- temp1[i, i]
		ttL[i] = sum(temp1[:, i]) .- temp1[i, i] 
		DCRL[i]= sum(temp1[i, :])
   		DCTL[i]= sum(temp1[:, i])
   		DCRLnorm[i]= sum(temp1[i, :] ./ temp1[i, i])
   		DCTLnorm[i]= sum(temp1[:, i] ./ temp1[i, i])

		trM[i] = sum(temp2[i, :]) .- temp2[i, i]
		ttM[i] = sum(temp2[:, i]) .- temp2[i, i] 
		DCRM[i]= sum(temp2[i, :])
   		DCTM[i]= sum(temp2[:, i])
   		DCRMnorm[i]= sum(temp2[i, :] ./ temp2[i, i])
   		DCTMnorm[i]= sum(temp2[:, i] ./ temp2[i, i])

		trS[i] = sum(temp3[i, :]) .- temp3[i, i]
		ttS[i] = sum(temp3[:, i]) .- temp3[i, i] 
		DCRS[i]= sum(temp3[i, :])
   		DCTS[i]= sum(temp3[:, i])
   		DCRSnorm[i]= sum(temp3[i, :] ./ temp3[i, i])
   		DCTSnorm[i]= sum(temp3[:, i] ./ temp3[i, i])

		trT[i] = sum(thetainf[i, :]) .- thetainf[i, i]
		ttT[i] = sum(thetainf[:, i]) .- thetainf[i, i] 
		DCRT[i]= sum(thetainf[i, :])
   		DCTT[i]= sum(thetainf[:, i])
   		DCRTnorm[i]= sum(thetainf[i, :] ./ thetainf[i, i])
   		DCTTnorm[i]= sum(thetainf[:, i] ./ thetainf[i, i])
	end
	NDCL = ttL .- trL
	NDCM = ttM .- trM
	NDCS = ttS .- trS
	NDCT = ttT .- trT

	# Connectedness measures
	WC1 = 100.0 .* (1 .- tr(temp1) ./ sum(temp1));
	TC1 = WC1 .* (sum(temp1) ./ sum(thetainf)); 

	WC2 = 100 .* (1 .- tr(temp2) ./ sum(temp2));
	TC2 = WC2 .* (sum(temp2) ./ sum(thetainf));

	WC3 = 100 .* (1 .- tr(temp3) ./ sum(temp3));
	TC3 = WC3 .* (sum(temp3) ./ sum(thetainf));

	#     % Total Frequency Connect
	TFC = TC1 .+ TC2 .+ TC3;

	return(TFC, TC1, TC2, TC3, WC1, WC2, WC3, DCTL, DCTM, DCTS, DCTT, DCRL, DCRM, DCRS, DCRT, DCTLnorm, DCTMnorm, DCTSnorm, DCTTnorm, DCRLnorm, DCRMnorm, DCRSnorm, DCRTnorm, NDCL, NDCM, NDCS, NDCT)
end


	
function f_all(it, ij, cut1, cut2,T::Int,  N::Int, L::Int, weights1, priorprec0, X, y, SI, PI, a, RI,corr)

	bayesalpha, bayessv, bayesgamma, BB = oneTimePrior(it, weights1, priorprec0, X, y, SI, PI, a, RI)

	HO = 100 + 1 # IRF Horizon, Also use this for FEVD horizon.
	ND = 1
	HORZ = HO - 1

	B0, A0 = getCoefsStable(bayesalpha, bayesgamma, bayessv, BB, N, L)

	wold = get_GIRF_fast(B0, A0, ND, N, L, HORZ)
	tfc, tcl, tcm, tcs, wcl, wcm, wcs, dctl, dctm, dcts, dctt, dcrl, dcrm, dcrs, dcrt, dctlnorm, dctmnorm, dctsnorm, dcttnorm, dcrlnorm, dcrmnorm, dcrsnorm, dcrtnorm, ndcl, ndcm, ndcs, ndct = get_dynnet(wold, T, A0,corr,cut1,cut2);
	return [[tfc, tcl, tcm, tcs, wcl, wcm, wcs]; dctl; dctm; dcts; dctt; dcrl; dcrm; dcrs; dcrt; dctlnorm; dctmnorm; dctsnorm; dcttnorm; dcrlnorm; dcrmnorm; dcrsnorm; dcrtnorm; ndcl; ndcm; ndcs; ndct]
end


function DynNet(data,cut1::Int,cut2::Int,L::Int,H::Int,Nsim::Int,corr)

	shrinkage = 0.05

	T, N = size(data)
	K = N * L + 1

	X = zeros(Float64, T - L, K-1)
	for i in 1:L
		temp = lag0(data, i)
	    X[:, (1 + N*(i-1) : i*N)] = temp[(1+L:T), :]
	end
	y = data[(1+L):T, :]
	T = T - L
	X = [ones(Float64, T, 1) X]

	K = N * L + 1

	SI, PI, a, RI = MinnNWprior(data, T, N, L, shrinkage)
	weights1 = convert.(Float64, normker(T, H))
	priorprec0 = convert.(Float64, inv(PI));

	xmean=zeros(7 + (2*4*2+4)*N,T);
	xci1=zeros(7 + (2*4*2+4)*N,T);
    xci2=zeros(7 + (2*4*2+4)*N,T);

	for it=1:T
    
	    out = zeros(7 + (2*4*2+4)*N,Nsim)
	    for i in 1:Nsim
	        out[:,i]=f_all(it, i, cut1,cut2,T, N, L, weights1, priorprec0, X, y, SI, PI, a, RI,corr)
	    end

	    #xmean[:,it] = mean(out,dims=2)
	    #xsd[:,it] = std(out,dims=2)
        
        xmean[:,it] = [quantile(out[i,:],0.5) for i=1:size(out)[1]]
	    xci1[:,it] = [quantile(out[i,:],0.025) for i=1:size(out)[1]]
        xci2[:,it] = [quantile(out[i,:],0.975) for i=1:size(out)[1]]
        
	end

	return(xmean,xci1,xci2)

end



function f_time(it, ij, T::Int,  N::Int, L::Int, weights1, priorprec0, X, y, SI, PI, a, RI,HH,corr)

	bayesalpha, bayessv, bayesgamma, BB = oneTimePrior(it, weights1, priorprec0, X, y, SI, PI, a, RI)

	HO = HH + 1 # IRF Horizon, Also use this for FEVD horizon.
	ND = 1
	HORZ = HO - 1

	B0, A0 = getCoefsStable(bayesalpha, bayesgamma, bayessv, BB, N, L)

	irf, = get_GIRF(B0, A0, ND, N, L, HORZ,corr)
	timecon, DCTT, DCRT, DCRTnorm, DCTTnorm, NDCT = get_timenet(N, HO, irf);
	
	return [timecon;DCTT;DCRT;DCRTnorm;DCTTnorm;NDCT]
end


function DynNet_time(data,L::Int,HH::Int,H::Int,Nsim::Int,corr)

	# L - VAR lags 
	# H - bandwidth width
	# Nsim - no of simulations

	shrinkage = 0.05

	T, N = size(data)
	K = N * L + 1

	X = zeros(Float64, T - L, K-1)
	for i in 1:L
		temp = lag0(data, i)
	    X[:, (1 + N*(i-1) : i*N)] = temp[(1+L:T), :]
	end
	y = data[(1+L):T, :]
	T = T - L
	X = [ones(Float64, T, 1) X]

	K = N * L + 1

	SI, PI, a, RI = MinnNWprior(data, T, N, L, shrinkage)
	weights1 = convert.(Float64, normker(T, H))
	priorprec0 = convert.(Float64, inv(PI))

	xmean=zeros(1+5*N,T);
	xci1=zeros(1+5*N,T);
    xci2=zeros(1+5*N,T);

	for it=1:T
    
	    out = zeros(1+5*N,Nsim)
	    for i in 1:Nsim
	        out[:,i]=f_time(it, i, T, N, L, weights1, priorprec0, X, y, SI, PI, a, RI,HH,corr)
	    end

	    #xmean[:,it] = mean(out,dims=2)
	    #xsd[:,it] = std(out,dims=2)
        
        xmean[:,it] = [quantile(out[i,:],0.5) for i=1:size(out)[1]]
	    xci1[:,it] = [quantile(out[i,:],0.025) for i=1:size(out)[1]]
        xci2[:,it] = [quantile(out[i,:],0.975) for i=1:size(out)[1]]
        
	end

	return(xmean,xci1,xci2)

end




##### GET dynamic tables


function DynNet_table(data,it,cut1::Int,cut2::Int,L::Int,H::Int,Nsim::Int,corr)

	shrinkage = 0.05

	T, N = size(data)
	K = N * L + 1

	X = zeros(Float64, T - L, K-1)
	for i in 1:L
		temp = lag0(data, i)
	    X[:, (1 + N*(i-1) : i*N)] = temp[(1+L:T), :]
	end
	y = data[(1+L):T, :]
	T = T - L
	X = [ones(Float64, T, 1) X]

	K = N * L + 1

	SI, PI, a, RI = MinnNWprior(data, T, N, L, shrinkage)
	weights1 = convert.(Float64, normker(T, H))
	priorprec0 = convert.(Float64, inv(PI));

	xmean=zeros(7 + (2*4*2+4)*N,T);
	xci1=zeros(7 + (2*4*2+4)*N,T);
    xci2=zeros(7 + (2*4*2+4)*N,T);

    out = [f_all_table(it, i, cut1,cut2,T, N, L, weights1, priorprec0, X, y, SI, PI, a, RI,corr) for i in 1:Nsim]

    table_all=[mean(filter(!isnan,[out[i][1][k,j] for i=1:Nsim])) for k=1:N, j=1:N]
	table_long=[mean(filter(!isnan,[out[i][2][k,j] for i=1:Nsim])) for k=1:N, j=1:N]
	table_medium=[mean(filter(!isnan,[out[i][3][k,j] for i=1:Nsim])) for k=1:N, j=1:N]
	table_short=[mean(filter(!isnan,[out[i][4][k,j] for i=1:Nsim])) for k=1:N, j=1:N]
       
	return(table_all,table_long,table_medium,table_short)

end


function f_all_table(it, ij, cut1, cut2,T::Int,  N::Int, L::Int, weights1, priorprec0, X, y, SI, PI, a, RI,corr)

	bayesalpha, bayessv, bayesgamma, BB = oneTimePrior(it, weights1, priorprec0, X, y, SI, PI, a, RI)

	HO = 100 + 1 # IRF Horizon, Also use this for FEVD horizon.
	ND = 1
	HORZ = HO - 1

	B0, A0 = getCoefsStable(bayesalpha, bayesgamma, bayessv, BB, N, L)

	wold = get_GIRF_fast(B0, A0, ND, N, L, HORZ)
	thetainf,temp1, temp2, temp3 = get_dynnet_table(wold, T, A0,corr,cut1,cut2);
	return (thetainf,temp1, temp2, temp3)
	
end



function get_dynnet_table(wo, TT, sig,corr,cut1::Int,cut2::Int)

	Tw = 200 # Define frequency window;
	omeg = LinRange(0.0, pi, Tw) # create equally spaced line from 0 to pi in 261 intervals

	# Define bands
	omeg2 = pi ./ omeg; 
	d1 = omeg2 .> cut2       # long term equals (20,260+] days
	d2 = (omeg2 .<= cut2) .* (omeg2 .> cut1)  # medium term equals (5,20] days
	d3 = omeg2 .<= cut1    # short term equals [1,5] days

	N = size(wo, 1)
	HO = size(wo, 3)

	if corr == true
		diag_sig = Diagonal(sig);
	end 
	
	if corr == false
		diag_sig = sig;
	end 
	
	expnnom = exp.(-im .* repeat(omeg, 1, HO) .* repeat((1:HO)', Tw, 1));
	expnnom = convert.(Complex{Float32}, expnnom)

	Omeg2 = Array{Float64}(undef, N, N)
	@views @inbounds for hh in 1:HO
		Omeg2 += wo[:,:,hh] * (diag_sig * wo[:,:,hh]')
	end
	Omeg2 = diag(Omeg2)

	wo = convert.(Float32, wo);
	FC = Array{Float64}(undef, N, N, Tw);

	GI = Array{Complex{Float32}}(undef, N, N);

	@views @inbounds for w in 1:Tw
		
		fill!(GI, 0.0);
		for nn in 1:HO
			GI .+= wo[:, :, nn] .* expnnom[w, nn];
		end

		PS = abs2.(GI * diag_sig); 

		@inbounds for k in 1:N 
			@inbounds for j in 1:N 
				FC[j, k, w] = PS[j, k] ./ (Omeg2[j] .* sig[k, k]);
			end
		end
	end

	PP1 = dropdims(sum(FC, dims = 3), dims=3);
	
	@views for w in 1:Tw
		for j in 1:N
			FC[j, :, w] = FC[j, :, w] ./ sum(PP1[j, :]); 
		end
	end

	thetainf = dropdims(sum(FC, dims = 3), dims=3);

	### BANDS : d1 d2 d3, Long, Medium, Short
	# theta_{d_i} summed over bands
	temp1 = dropdims(sum(FC[:, :, d1], dims = 3), dims=3); 
	temp2 = dropdims(sum(FC[:, :, d2], dims = 3), dims=3);
	temp3 = dropdims(sum(FC[:, :, d3], dims = 3), dims=3);

	for j in 1:N
		sumthetaj = sum(thetainf[j, :]);
		temp1[j, :] = temp1[j, :] ./ sumthetaj;
		temp2[j, :] = temp2[j, :] ./ sumthetaj;
		temp3[j, :] = temp3[j, :] ./ sumthetaj;
	end

	return(thetainf,temp1, temp2, temp3)
end


