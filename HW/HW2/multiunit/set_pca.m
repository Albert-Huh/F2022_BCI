function set_pca(newval)


global parms;

% Valid value?
if(newval < 0 | newval > size(parms.patterns.MI{1}, 2)) 
  strg = sprintf('You can only select a pca compression level between 0 and %d\n  (with 0 == no compression)', ...
		size(parms.patterns.MI{1}, 2));
  disp(strg);
  strg
  error(' ');
end;

if(parms.pca.ncomponents == 0 & newval ~= 0)
  % Force reconfiguration of the training inputs (including pca computation)
  parms.net.patset_reset_flag = 1;
end;

parms.pca.ncomponents = newval;
parms.net.pca_reset_flag = 1;

