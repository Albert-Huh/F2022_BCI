function status

% Show the configuration of the learning system

global parms;

disp('Folds available: ');
folds = 1:parms.patterns.Nfolds;
disp(folds);

disp(sprintf('\nTraining set:'));
disp(parms.net.patset_train);

disp(sprintf('\nTest set:'));
disp(parms.net.patset_test);

if(length(intersect(parms.net.patset_train, parms.net.patset_test)) > 0)
  disp('*** WARNING: test and training sets overlap ***');
end;

disp(sprintf('\nDelays: '));
disp(parms.patterns.elements_base);

disp(sprintf('\nPCA compression: %d', parms.pca.ncomponents));

disp(sprintf('\nProprioceptive append flag: %d', parms.patterns.append_flag));

disp(' ');
disp('Output type:');
disp(['    ', parms.patterns.type]);
disp(' ');

if(parms.net.train_flag)
  disp('Training Fraction of Variance Accounted For (last training):');
  disp(parms.net.fvaf_train);

  disp('Test Fraction of Variance Accounted For (last training): ');
  disp(parms.net.fvaf_test);
end;





