function figure_in(fig)

% Shift figure to the left

figure(fig);

fparms = get(gcf, 'UserData');

fparms.width = fparms.width/2;
fparms.start = fparms.start + fparms.width/2;

set(gcf, 'UserData', fparms);

gen_figure(fparms);

