global parms;

parms.patterns.Nfolds = 10;
parms.patterns.fbase = '../motorcontrol/files_2005_07';
parms.pca.ncomponents = 0;

parms.net.patset_train = [1];
parms.net.patset_test = [10];
parms.net.patset_reset_flag = 1;
parms.net.outset_reset_flag = 1;
parms.net.pca_reset_flag = 1;

parms.patterns.types = {'torque', 'X', 'dX', 'theta', 'dtheta', 'ddX', 'ddtheta'};
parms.patterns.type = parms.patterns.types{1};

parms.fig.types = {'train', 'test', 'input_train', 'input_test'};
parms.fig.showelems = [1:100];
parms.patterns.dt = .05;

parms.patterns.append_flag = 0;

parms.net.train_flag = 0;

%%% Arm length parms (Raju)
parms.arm.L1 = .139;
parms.arm.L2 = .204;
parms.arm.types = {'train', 'test'};

%%
parms.patterns.Ndelays = 20;


% Load in data
for j=1:parms.patterns.Nfolds
  strg = sprintf('%s/MI_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  strg
  parms.patterns.MI{j} = load(strg);

  strg = sprintf('%s/X_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.X{j} = load(strg);

  strg = sprintf('%s/torque_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.torque{j} = load(strg);

  strg = sprintf('%s/theta_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.theta{j} = load(strg);

  strg = sprintf('%s/dtheta_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.dtheta{j} = load(strg);

  strg = sprintf('%s/dX_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.dX{j} = load(strg);

  strg = sprintf('%s/ddX_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.ddX{j} = load(strg);

  strg = sprintf('%s/ddtheta_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.ddtheta{j} = load(strg);

  strg = sprintf('%s/time_fold_DAT6_08_%02d.txt', parms.patterns.fbase, j);
  parms.patterns.time{j} = load(strg);

end;

set_delays(1:parms.patterns.Ndelays);
