clear
close all

load fit_sim.mat


rtCor = rt(resp>0);
[N,X] = hist(rtCor,nbins(rtCor)+10);
dx = X(2)-X(1);
NN = cumsum(N);
acc = sum(resp>0)/length(resp);
NN = NN/NN(end)*acc;
%bar(X,NN,'b')
figure(1)
bar(X,N/trapz(X,N),'b')

%% Empirical distribution function
figure(2)
[ff,xx,flo,fup] = ecdf(rtCor,'alpha',.05,'bounds','on');
h1 = stairs(xx,ff,'linewidth',1);


%% Run msddm code, quickest just to type these in, but you can look
%% at recoveredParameters variable
a = [0.2132 0.5898];
s = [1 1];        
th = [1.4987 1.995];
x0dist = 1;        
dl = [0 1.4997];
tFinal = 13;
x0 = -0.1133;
t0 = 0;
[tArray,~,yPlus,yMinus] = multistage_ddm_fpt_dist(...
    a,s,th,x0,x0dist,dl,tFinal);
figure(2)
hold on
h2 = plot(tArray+t0,yPlus/yPlus(end),'r','Linewidth',2);
stairs(xx,flo,'b:','linewidth',1);
stairs(xx,fup,'b:','linewidth',1);

for jj = 1:length(yPlus)-1
    yy(jj) = (yPlus(jj+1)-yPlus(jj))/(tArray(jj+1)-tArray(jj));
end

% dt = tArray(3)-tArray(2);
 tt = tArray(2:end);
% yy = diff(yPlus)/dt;
figure(1)
hold on
plot(tt+t0,yy/trapz(tt,yy),'r','Linewidth',2)
drawnow


figure(1)
title('Simulated RT histogram and model fit','FontSize',18,'FontWeight','bold')
xlabel('Reaction time (sec)','FontSize',18)
ylabel('Frequency','FontSize',18)
%legend('Experimental','2-stage DDM','FontSize',18)
saveas(gcf,'figures/rt_simfit.eps','psc2')
saveas(gcf,'figures/rt_simfit.fig','fig')

figure(2)
title('Simulated RT CDF and model fit','FontSize',18,'FontWeight','bold')
xlabel('Reaction time (sec)','FontSize',18)
ylabel('','FontSize',18)
%legend([h1 h2],'Empirical distribution function','2-stage DDM CDF','Location','NorthWest')
saveas(gcf,'figures/rtDF_simfit.eps','psc2')
saveas(gcf,'figures/rtDF_simfit.fig','fig')

return

