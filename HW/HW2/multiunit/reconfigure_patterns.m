function reconfigure_patterns

global parms;

if(parms.net.patset_reset_flag)

  parms.patterns.input_ftrain = [];
  % Construct training set
  for k=1:length(parms.net.patset_train)
    tmp = parms.patterns.MI{parms.net.patset_train(k)};
    if(parms.patterns.append_flag)
      % Cell + theta + dtheta
      parms.patterns.input_ftrain = [parms.patterns.input_ftrain; ...
		    [tmp(:,parms.patterns.elements), ...
		     parms.patterns.theta{parms.net.patset_train(k)}, ...
		     parms.patterns.dtheta{parms.net.patset_train(k)} ...
		    ]];
    else
      % Cell only
      parms.patterns.input_ftrain = [parms.patterns.input_ftrain; ...
		    tmp(:,parms.patterns.elements)];
      end;
  end;
  
  % Construct test set
  parms.patterns.input_ftest = [];
  for k=1:length(parms.net.patset_test)
    tmp = parms.patterns.MI{parms.net.patset_test(k)};
    if(parms.patterns.append_flag)
      % Cell + theta + dtheta
      parms.patterns.input_ftest = [parms.patterns.input_ftest; ...
		    [tmp(:,parms.patterns.elements), ...
		     parms.patterns.theta{parms.net.patset_test(k)}, ...
		     parms.patterns.dtheta{parms.net.patset_test(k)} ...
		    ]];
    else
      % Cell only
      parms.patterns.input_ftest = [parms.patterns.input_ftest; ...
		    tmp(:,parms.patterns.elements)];
    end;
  end;

  if(parms.pca.ncomponents ~= 0)
    disp('Computing principal components...');
    % Compute principal components of training set
    %  [parms.pca.components, explained, parms.pca.tr_mean] = ...
    %      compute_pc(parms.patterns.input_ftrain);
    [parms.pca.transMat, parms.pca.meanp, parms.pca.stdp] = ...
	compute_pc(parms.patterns.input_ftrain);
  end;
end;

% Need to reproject into the principal components?
if(parms.net.pca_reset_flag | parms.net.patset_reset_flag)
  % Yes
  if(parms.pca.ncomponents > 0)
    % Apply PCA compression to the training set
    disp('Compressing training set...');
%    parms.patterns.input_train = compress2(parms.pca.components, ...
%					 parms.patterns.input_ftrain, ...
%					 parms.pca.ncomponents, ...
%					 parms.pca.tr_mean);
    parms.patterns.input_train = compress2(parms.pca.transMat, ...
					 parms.patterns.input_ftrain, ...
					 parms.pca.ncomponents, ...
					   parms.pca.meanp, ...
					   parms.pca.stdp ...
					   );

    % Apply PCA compression to the test set
    disp('Compressing test set...');
%    parms.patterns.input_test = compress2(parms.pca.components, ...
%					 parms.patterns.input_ftest, ...
%					 parms.pca.ncomponents, ...
%					 parms.pca.tr_mean);
    parms.patterns.input_test = compress2(parms.pca.transMat, ...
					 parms.patterns.input_ftest, ...
					 parms.pca.ncomponents, ...
					   parms.pca.meanp, ...
					   parms.pca.stdp ...
					   );
    disp('Done.');
    
  else
    % No compression
    disp('No PCA compression...');
    parms.patterns.input_train = parms.patterns.input_ftrain;
    parms.patterns.input_test = parms.patterns.input_ftest;
  end;
  
  % Append 1's
  parms.patterns.input_train = [parms.patterns.input_train, ...
		    ones(size(parms.patterns.input_train,1),1)];
  
  parms.patterns.input_test = [parms.patterns.input_test, ...
		    ones(size(parms.patterns.input_test,1),1)];

  % Statistics for display
  parms.patterns.input_train_min = min(parms.patterns.input_train);
  parms.patterns.input_train_max = max(parms.patterns.input_train);
  parms.patterns.input_train_range = max(parms.patterns.input_train_max - ...
					 parms.patterns.input_train_min, .001);
  parms.patterns.input_test_min = min(parms.patterns.input_test);
  parms.patterns.input_test_max = max(parms.patterns.input_test);
  parms.patterns.input_test_range = max(parms.patterns.input_test_max - ...
					 parms.patterns.input_test_min, .001);

  
end;

% Need to reconfigure the output side?
if(parms.net.outset_reset_flag | parms.net.pca_reset_flag ...
   | parms.net.patset_reset_flag)
  disp('Reconfiguring output side...');
  % Construct training set
  parms.patterns.output_train = [];
  parms.patterns.time_train = [];
  for k=1:length(parms.net.patset_train)
    switch(strmatch(parms.patterns.type, parms.patterns.types, 'exact'))
      case 1
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.torque{parms.net.patset_train(k)}];
     case 2
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.X{parms.net.patset_train(k)}];
     case 3
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.dX{parms.net.patset_train(k)}];
     case 4
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.theta{parms.net.patset_train(k)}];
     case 5
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.dtheta{parms.net.patset_train(k)}];
     case 6
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.ddX{parms.net.patset_train(k)}];
     case 7
       parms.patterns.output_train = [parms.patterns.output_train; ...
		    parms.patterns.ddtheta{parms.net.patset_train(k)}];
     otherwise
       error('Invalid type');
    end;
    
    parms.patterns.time_train = [parms.patterns.time_train; ...
		    parms.patterns.time{parms.net.patset_train(k)}];

  end;
  
  % Construct test set
  parms.patterns.output_test = [];
  parms.patterns.time_test = [];
  for k=1:length(parms.net.patset_test)
    switch(strmatch(parms.patterns.type, parms.patterns.types, 'exact'))
     case 1
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.torque{parms.net.patset_test(k)}];
      
     case 2
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.X{parms.net.patset_test(k)}];
      
     case 3
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.dX{parms.net.patset_test(k)}];
      
     case 4
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.theta{parms.net.patset_test(k)}];
      
     case 5
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.dtheta{parms.net.patset_test(k)}];
      
     case 6
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.ddX{parms.net.patset_test(k)}];
      
     case 7
      parms.patterns.output_test = [parms.patterns.output_test; ...
		    parms.patterns.ddtheta{parms.net.patset_test(k)}];
      
     otherwise
      error('Invalid type');
    end;
    parms.patterns.time_test = [parms.patterns.time_test; ...
		    parms.patterns.time{parms.net.patset_test(k)}];
      
  end;
end;
  


  
parms.net.patset_reset_flag = 0;
parms.net.outset_reset_flag = 0;
