function animate(mode)

% Show animation of arm movement

global parms;

% Check usage
if(nargin == 0)
  disp('Usage: animate(mode)');
  disp(' Valid modes include:');
  for j=1:length(parms.arm.types);
    disp(['   ',  parms.arm.types{j}]);
  end;
  return;
end;

% Make sure that we are predicting theta
if(~strcmp(parms.patterns.type, 'theta'))
  disp('animate() only works if you are predicting joint (choose type theta).');
  return;
end;

% Make sure that we have actually trained ...
if(parms.net.outset_reset_flag)
  disp('You must train first');
  return;
end;


% Figure number
fig = 100;

% Find the requested type in the list of valid types
str = strmatch(mode, parms.arm.types);

if(isempty(str))
  disp('Valid types include:');
  for j=1:length(parms.arm.types);
    disp(['   ',  parms.arm.types{j}]);
  end;
  error('Please choose a valid type');
end;

% Pull out the data that we want (train or test)
if(str == 1)
  dat_pred = parms.net.output_train;
  dat_act = parms.patterns.output_train;
else
  dat_pred = parms.net.output_test;
  dat_act = parms.patterns.output_test;
end;

% Generate the figure
parms.arm.halt_flag = 0;

figure(fig);
set(gcf, 'DoubleBuffer', 'on');
h = plot([0; -.2; -.2; 0; 0]/2,[0;0;-.1;-.1;0]/2, 'k');
set(h, 'linewidth', 2);
xlim([-.3,.35]);
ylim([-.2,.45]);

% Stop button
uicontrol('Style', 'pushbutton', ...
	  'String', 'stop', ...
	  'units', 'normalized', ...
	  'Position', [.01, .925 .1 .05], ...
	  'Callback', ...
	  ['parms.arm.halt_flag = 1;']);


% Iterate through the frames
for j=1:size(dat_act,1)
  % Actual
  X = arm_kinematics(dat_act(j,:));
  obj_act = line('Xdata', [0,X(1,:)], 'ydata', [0,X(2,:)], 'linewidth', ...
	                4);
  set(obj_act, 'color', [0, 0, 1]);
  
  % Predicted
  X = arm_kinematics(dat_pred(j,:));
  obj_pred = line('Xdata', [0,X(1,:)], 'ydata', [0,X(2,:)], 'linewidth', ...
	                4);
  set(obj_pred, 'color', [1, 0, 0]);

  % Force a drawing
  drawnow;

  % Should we halt?
  if(parms.arm.halt_flag)
    close(fig);
    return;
  end;
  
  pause(.025);

  % Delete old objects so that we can create new ones
  delete(obj_act);
  delete(obj_pred);
end;

% Clean up
close(fig);


