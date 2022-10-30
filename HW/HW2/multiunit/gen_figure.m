function gen_figure(fparms)

% 
global parms;
%fparms
range = fparms.start + (1:fparms.width)-1;
tm = range * parms.patterns.dt;

switch fparms.type
  case 1
   subplot(2,1,1);
   plot(tm, parms.patterns.output_train(range,1));
   hold on;
   plot(tm, parms.net.output_train(range,1), 'r');
   hold off;
   title('Output (Training Set)');
   %ylabel([parms.patterns.types{1}, ', dim 1']);
   ylabel([parms.patterns.type, ', dim 1']);
   subplot(2,1,2);
   plot(tm, parms.patterns.output_train(range,2));
   hold on;
   plot(tm, parms.net.output_train(range,2), 'r');
   ylabel([parms.patterns.type, ', dim 2']);
   hold off;
 case 2
   subplot(2,1,1);
   plot(tm, parms.patterns.output_test(range,1));
   hold on;
   plot(tm, parms.net.output_test(range,1), 'r');
   hold off;
   ylabel([parms.patterns.type, ', dim 1']);
   title('Output (Test Set)');
   subplot(2,1,2);
   plot(tm, parms.patterns.output_test(range,2));
   hold on;
   plot(tm, parms.net.output_test(range,2), 'r');
   ylabel([parms.patterns.type, ', dim 2']);
   hold off;
 case 3
  dat = (parms.patterns.input_train - ...
	 ones(size(parms.patterns.input_train,1),1) * parms.patterns.input_train_min) ...
	./ (ones(size(parms.patterns.input_train,1),1) *  ...
	    (parms.patterns.input_train_range));
  dat = dat(:, intersect(parms.fig.showelems, 1:size(dat,2)));
  dat = dat + (ones(size(dat,1),1) * (1:size(dat,2)));
  plot(tm, dat(range,:));
  title('Input (Training Set)');
  ylabel('Input Index');
 case 4
  dat = (parms.patterns.input_test - ...
	 ones(size(parms.patterns.input_test,1),1) * parms.patterns.input_test_min) ...
	./ (ones(size(parms.patterns.input_test,1),1) *  ...
	    (parms.patterns.input_test_range));
  dat = dat(:, intersect(parms.fig.showelems, 1:size(dat,2)));
  dat = dat + (ones(size(dat,1),1) * (1:size(dat,2)));
  plot(tm, dat(range,:));
  title('Input (Test Set)');
  ylabel('Input Index');
end

xlabel('time (sec)');
