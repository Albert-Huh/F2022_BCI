range = 1:300;
figure(1);
plot(parms.patterns.time_train(range,1), parms.net.output_train(range,1), 'b');
hold on;
plot(parms.patterns.time_train(range,1), parms.patterns.output_train(range,1), 'r');
hold off;



figure(2);
plot(parms.patterns.time_test(range,1), parms.net.output_test(range,1), 'b');
hold on;
plot(parms.patterns.time_test(range,1), parms.patterns.output_test(range,1), 'r');
hold off;



