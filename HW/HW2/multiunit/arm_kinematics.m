function X = arm_kinematics(theta)

global parms;

   c1 = cos(theta(1));
   s1 = sin(theta(1));
   c2 = cos(theta(2));
   s2 = sin(theta(2));
   c12 = cos(theta(1) + theta(2));
   s12 = sin(theta(1) + theta(2));
  
   % Elbow position
   X(1,1) = parms.arm.L1 * c1;
   X(2,1) = parms.arm.L1 * s1;

   % Wrist position
   X(1,2) = X(1,1) + parms.arm.L2 * c12;
   X(2,2) = X(2,1) + parms.arm.L2 * s12;

   

