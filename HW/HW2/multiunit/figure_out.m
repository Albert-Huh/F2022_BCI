function figure_out(fig)

% Shift figure to the left

global parms;

figure(fig);

fparms = get(gcf, 'UserData');

% Check that the resulting width will fit the data
if(fparms.type == 1 | fparms.type == 3)
  % Training set display
  if(fparms.width*2 > size(parms.patterns.input_train,1))
    disp('Cant zoom out any more');
    return
  end;
  
else
  % Test set display  
  if(fparms.width*2 > size(parms.patterns.input_test,1))
    disp('Cant zoom out any more');
    return
  end;

end;

fparms.width = fparms.width*2;
fparms.start = fparms.start - fparms.width/2;

% Check start boundary
if(fparms.start < 1)
  fparms.start = 1;
end;

% Check end boundary
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

