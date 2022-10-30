function set_test(newset)

% Select the set of folds for testing

global parms;

% Values in range?
elems = find(newset < 1 | newset > parms.patterns.Nfolds);
if(length(elems) > 0)
  % Out of range
  error(['Can only select specify values between 1 and ', ...
	 num2str(parms.patterns.Nfolds)]);
end;

% Overlap with test set?
if(intersect(newset, parms.net.patset_train))
  disp('Warning: specified pattern set overlaps with test set.');
end;

% Ok - make the change

parms.net.patset_test = newset;
parms.net.patset_reset_flag = 1;
