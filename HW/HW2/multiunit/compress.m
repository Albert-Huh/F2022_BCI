function out = compress(pc, in, outdim, tr_mean)

% Reduce the dimensionality of <in> to <outdim>, using pc, and place the results
% in <out>.
in = in -   ones(size(in,1),1) * tr_mean;
out = in*pc;
out = out(:,1:outdim);
