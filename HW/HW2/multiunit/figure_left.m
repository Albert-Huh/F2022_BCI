function figure_left(fig)

% Shift figure to the left

figure(fig);

fparms = get(gcf, 'UserData');

fparms.start = fparms.start - fparms.width/2;
if(fparms.start < 1)
  fparms.start = 1;
end;

set(gcf, 'UserData', fparms);

gen_figure(fparms);

