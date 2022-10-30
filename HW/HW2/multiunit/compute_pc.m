function [transMat, meanp, stdp] = compute_pc(in)
%function [pc explained tr_mean] = compute_pc(in)

% Extract the principle components
%covx = cov(in);
%[pc,variances,explained] = pcacov(covx);
%tr_mean = mean(in,1);
in_trans = in';

[pn,meanp,stdp] = prestd(in_trans);
[ptrans, transMat] = prepca(in_trans, 0);

