function out = compress2(transMat, in, outdim, meanp, stdp)

% Reduce the dimensionality of <in> to <outdim>, using transMat, and place the results
% in <out>.
in2 = trastd(in', meanp, stdp);
out = trapca(in2, transMat(1:outdim, :))';
