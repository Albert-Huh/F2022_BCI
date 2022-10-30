function [Etr,Ete]=train

global parms;

reconfigure_patterns;

disp('Training...');
% Do the learning
parms.net.W = pinv(parms.patterns.input_train) * parms.patterns.output_train;

disp('Evaluating...');
% Now generate the train and test outputs
parms.net.output_train = parms.patterns.input_train * parms.net.W;

parms.net.output_test = parms.patterns.input_test * parms.net.W;

parms.net.fvaf_train = compute_fvaf(parms.patterns.output_train, ...
				    parms.net.output_train);

parms.net.fvaf_test = compute_fvaf(parms.patterns.output_test, ...
				    parms.net.output_test);
disp(' ');
disp('Output type:');
disp(['    ', parms.patterns.type]);
disp(' ');

disp('Training Fraction of Variance Accounted For:');
disp(parms.net.fvaf_train);

disp('Test Fraction of Variance Accounted For:');
disp(parms.net.fvaf_test);


parms.net.train_flag = 1;   % Remember that we have performed training
Etr=parms.net.fvaf_train;
Ete=parms.net.fvaf_test;