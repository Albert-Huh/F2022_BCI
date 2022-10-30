function figure_regen(fig)

% Redraw the figure

figure(fig);

fparms = get(gcf, 'UserData');

gen_figure(fparms);

