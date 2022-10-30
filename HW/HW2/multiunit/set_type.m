function set_type(newtype)

global parms;

if(nargin ~= 1)
  disp('Usage: set_type(type)');
  disp('   Valid types include:');
  for j=1:length(parms.patterns.types);
    disp(['   ',  parms.patterns.types{j}]);
  end;
  return;
end;

% Find the requested type in the list of valid types
str = strmatch(newtype, parms.patterns.types);


if(isempty(str))
  disp('Valid types include:');
  for j=1:length(parms.patterns.types);
    disp(['   ',  parms.patterns.types{j}]);
  end;
  error('Please choose a valid type');
end;

% Everything ok: remember the type
parms.patterns.type = newtype;

% Force reconfigure_patterns to update the patterns
parms.net.outset_reset_flag = 1;
