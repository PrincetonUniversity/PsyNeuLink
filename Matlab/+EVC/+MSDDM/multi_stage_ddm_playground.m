%%
clc
clear all;
import EVC.MSDDM.*;

a = [-0.3 1];
s = [0.5 0.65];
deadlines = [0 0.2];
thresholds = [1 2];
x0 = 0;
x0dist = 1;


a_vec = [-0.3 1; ...
         -0.2 1.5];
s_vec = [0.5 0.65; ...
         0.65 0.5];
deadlines_vec = [0 0.2; ...
                 0 0.3];
thresholds_vec = [1 2; ...
                  1.5 2.5];
x0_vec = [0; ...
          0.1];
x0dist_vec = [1; ...
              1];


[mean_RT, mean_ER, mean_RT_plus, mean_RT_minus] = multi_stage_ddm_metrics(a ,s, deadlines, thresholds, x0, x0dist);
[mean_RT_vec, mean_ER_vec, mean_RT_plus_vec, mean_RT_minus_vec] = multi_stage_ddm_metrics_VEC(a ,s, deadlines, thresholds, x0, x0dist);
