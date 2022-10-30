function set_proprioceptive(newval)


global parms;

if(newval ~= 0 & newval ~= 1)
  error('Value must be 0 or 1');
end;


if(newval ~= parms.patterns.append_flag)
  parms.patterns.append_flag = newval;
  parms.net.patset_reset_flag = 1;
  disp(['Proprioceptive flag set to ', num2str(newval)]);
else
  disp('No change');
end;

