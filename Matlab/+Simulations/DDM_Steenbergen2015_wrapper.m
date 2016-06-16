%%

function [params y] = DDM_Steenbergen2015_wrapper()
clear all;
import Simulations.*;

saliences = 0.42:0.002:0.5;
driftEfficiencies = 0.6:0.01:1.5;

params = driftEfficiencies;
y = zeros(1, length(params));

for i = 1:length(params)
    EVCSim = DDM_Steenbergen2015();
    EVCSim.printResults = 0;
    EVCSim.nSubj = 1;
    EVCSim.plotSum = 0;
    %EVCSim.stimSalience = params(i);
    EVCSim.DDMProcesses(1).input.params{3}.params{2} = params(i);
    EVCSim.run(); 
    y(i) = mean(EVCSim.results.RT.adaptEffect2);
end

%% plot
% figure(1);
% plot(params, y, '-k','LineWidth', 3);
% xlabel('task difficulty (drift efficiency)', 'FontSize',20);
% ylabel('conflict adaptation score (ms)', 'FontSize',20);
% ylim([-200 300]);
% %xlabel('task difficulty (saliency)', 'FontSize',20);
% set(gca,'FontSize',14)

end