function set_delays(delays)

% Set the delays at which we are looking at neural activity
%  set_delays(1) : 0 delay only
%  set_delays(1:2) : 0, 50ms delays
%  set_delays(1:3) : 0, 50ms, 100ms delays
%  set_delays(1:20) : 0 ... 950ms delays

global parms;

if(nargin == 0)
  disp('Usage: set_delays(vec)');
  disp(sprintf(' where vec consists of numbers 1 .. %d', parms.patterns.Ndelays));
  return;
end;

delays = intersect(delays, 1:parms.patterns.Ndelays);

if(length(delays) == 0)
  disp('Usage: set_delays(vec)');
  disp(sprintf(' where vec consists of numbers 1 .. %d', parms.patterns.Ndelays));
  return;
end;

elems = parms.patterns.Ndelays + 1 - delays;

Ncells = size(parms.patterns.MI{1},2) / parms.patterns.Ndelays;

accum = [];
for j = 1:Ncells
  accum = [accum, elems + parms.patterns.Ndelays * (j-1)];
end;
parms.patterns.elements = accum;
parms.patterns.elements_base = elems;

disp('ok');

parms.net.patset_reset_flag = 1;


