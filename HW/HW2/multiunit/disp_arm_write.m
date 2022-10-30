%  Display the arm on the current plot

function [obj] = disp_arm_write(X)

global disp;

if(ishandle(disp.arm_obj))
  delete(disp.arm_obj);
end;

obj = line('Xdata', [0,X(1,:)], 'ydata', [0,X(2,:)], 'linewidth', ...
	   4);
disp.arm_obj = obj;
set(obj, 'color', [0,1,1]);

