function figure_right(fig)

global parms;

% Shift figure to the left

figure(fig);

fparms = get(gcf, 'UserData');

fparms.start = fparms.start + fparms.width/2;

if(fparms.type == 1 | fparms.type == 3)
  % Training set display
  if(fparms.start+fparms.width > size(parms.patterns.input_train,1))
    fparms.start = size(parms.patterns.input_train,1) - fparms.width + 1;
  end;
  
else
  % Test set display  
  if(fparms.start+fparms.width > size(parms.patterns.input_test,1))
    fparms.start = size(parms.patterns.input_test,1) - fparms.width + 1;
  end;

end;

set(gcf, 'UserData', fparms);

gen_figure(fparms);


