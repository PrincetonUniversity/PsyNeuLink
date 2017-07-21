classdef ModelOfEVOC
    %The metalevel model represents the control system's knowledge about the 
    %values of alternative combinations of control signals.
    
    properties
        %measured (feature,run time, value)-triplets
        observations;
                
        %number of algorithms available to the agent
        nr_basic_features;
        nr_control_signals;
        signal_range=[0,1];
        
        max_nr_regressors;
        nr_regressors;
        extended_features;
        regressor_names;
        
        glm_EVOC;
        selected_regressors;        
                      
        time_cost;
        
        %control cost functions
        %  a) implementation cost: f(c)=exp(a*|c|+b)
        implementation_cost=struct('a',1/4, 'b',-3);
        %  b) reconfiguration cost: g(c)=exp(a*|c-c_prev|+b)
        reconfiguration_cost=struct('a',1/4,'b',-3);
        
    end
    
    methods

        %constructor
        function model=ModelOfEVOC(nr_features,nr_control_signals)
            
            model.nr_control_signals=nr_control_signals;            
            model.nr_regressors=(nr_features+1)*(nr_control_signals+1)+2;
            model.nr_basic_features=nr_features;
            %regressors: 1, features, control signals, interactions, cost 
            model.glm_EVOC=BayesianGLM(model.nr_regressors);

            
            model.observations=struct('features',cell(1,1),...
                'control_signals',cell(1,1),...
                'VOC',cell(1,1));
                                       
            model.selected_regressors=1:model.nr_regressors;
                        
        end
        
        %predict EVOC
        function [EVOC,sigma_VOC]=predictPerformance(model,problem_features,control_signals)
                        
            regressors=model.constructRegressors(problem_features,control_signals);            
                
            EVOC=dot(regressors,model.glm_EVOC.mu_n);
            sigma_VOC=sqrt(max(1e-9,model.glm_EVOC.b_n/model.glm_EVOC.a_n));
        end
        
        function VOC_samples=sampleVOC(model,...
                problem_features,control_signals,nr_samples)
            
            [EVOC,sigma_VOC]=model.predictPerformance(problem_features,control_signals);
                        
            VOC_samples=mvnrnd(EVOC,sigma_VOC,nr_samples);
                        
        end
        
        %learning
        function model=learn(model,control_signals,features,utility,cost)
                        
            model.observations.features=[model.observations.features;features];
            model.observations.control_signals=[model.observations.control_signals;control_signals];
            model.observations.VOC=[model.observations.VOC;utility-cost];
                
            %1. Learn to predict the VOC           
            regressors=model.constructRegressors(features,control_signals)';            
            model.glm_EVOC=model.glm_EVOC.update(regressors,utility-cost);

        end
        
        function c_star=selectControlSignal(model,features,nr_samples)
            %features: basic features of the problem
            nr_features=numel(features);
            
            if not(exist('nr_samples','var'))
                nr_samples=1;
            end
            
            %1. Find Optimal Control Signal Vector by gradient ascent
            eta=1;
            for s=1:nr_samples
                [~,samples(:,s),~]=model.glm_EVOC.samplePosterior();
            end
            sampled_coefficients=mean(samples,2);
            
            
            first_ind=(nr_features+model.nr_control_signals+2);
            last_ind=nr_features*model.nr_control_signals+nr_features+...
                model.nr_control_signals+1;
            interaction_coefficients=reshape(sampled_coefficients(first_ind:last_ind),...
                [model.nr_control_signals,nr_features]);
            cost_coefficients=sampled_coefficients(end-1:end);
            signal_coefficients=sampled_coefficients(...
                (nr_features+2):(nr_features+model.nr_control_signals+1),1);
                        
            dQ_dc=@(c) signal_coefficients+cost_coefficients(1)*model.implementationCostGradient(c)+...
                cost_coefficients(2)*model.reconfigurationCostGradient(c)+...
                interaction_coefficients*features;
            
            delta_c=inf;
            epsilon=0.025;
            c_old=model.previousControlSignal();
            
            i=1;
            while delta_c>epsilon
                c_new=max(0,c_old+eta/sqrt(i)*dQ_dc(c_old));
                delta_c=norm(c_new-c_old,2)/max(1,norm(c_old));
                c_old=c_new;
                i=i+1;
            end
            
            c_star=c_new;
        end
        
        %{
        function model=modelSelection(model,a)
            %a: algorithm for which to update the model selection
            
            %{
            %possible models
            possible_models=nchoose(1:model.nr_regressors);
            nr_possible_models=length(possible_models);
            
            
            if isempty(model.observations(a).features)
                p_m(:,a)=ones(nr_possible_models,1)/nr_possible_models;
                return;
            end
            
            regressors=model.constructRegressors(...
                model.observations(a).features);
            
            %functional form of the expected run time for algorithm a
            for m=1:nr_possible_models
                %q_T=tapas_vblm(model.observations(a).run_time,regressors(:,possible_models{m}),...
                %    model.a0_alpha_mu_runtime, model.b0_alpha_mu_runtime,...
                %    model.alpha0_epsilon_T, model.beta0_epsilon_T);
                
                glm=BayesianGLM(length(possible_models{m}));
                log_p_y(m)=glm.logMarginalLikelihood(regressors(:,possible_models{m}),...
                    model.observations(a).run_time);
            end
            
            p_m(:,a)=exp(log_p_y-logsumexp(log_p_y));
            [max_log_p(a),best_model(a)]=max(p_m(:,a));
            %}
            model.glm_runtime{a}=BayesianGLM(model.max_nr_regressors);
            regressors=model.constructRegressors(model.observations(a).features);
            y=model.observations(a).run_time;
            [model.selected_regressors_runtime{a},p_m]=...
                model.glm_runtime{a}.featureSelection(regressors,y);
            
            
            model.glm_runtime{a}=BayesianGLM(numel(model.selected_regressors_runtime{a}));
            model.glm_runtime{a}=model.glm_runtime{a}.learn(regressors(:,model.selected_regressors_runtime{a}),...
                model.observations(a).run_time);
            
        end
        %}
        
        function [regressors,model]=constructRegressors(model,problem_features,control_signals)
                        
            feature_signal_interactions=kron(control_signals,problem_features);
            regressors=[1;problem_features;control_signals;feature_signal_interactions;...
                model.implementationCost(control_signals); ...
                model.reconfigurationCost(control_signals)];
            
        end
        
        function cost=implementationCost(model,control_signal)
            cost=exp(model.implementation_cost.a*norm(control_signal,2)+...
                model.implementation_cost.b);
        end
        
        function cost=reconfigurationCost(model,control_signal)
            cost=exp(model.reconfiguration_cost.a*norm(control_signal,2)+...
                model.reconfiguration_cost.b);
        end
        
        function cost_gradient=implementationCostGradient(model,control_signals)
            if norm(control_signals,2)>0
                cost_gradient=model.implementation_cost.a/(eps+norm(control_signals,2))*...
                    model.implementationCost(control_signals)*(control_signals+eps);
            else
                cost_gradient=zeros(model.nr_control_signals,1);
            end                           
        end
        
        function cost_gradient=reconfigurationCostGradient(model,control_signals)
            
            prev_control_signals=model.previousControlSignal();
            
            if norm(control_signals-prev_control_signals,2)>0
                cost_gradient=model.reconfiguration_cost.a/norm(control_signals-prev_control_signals,2)*...
                    model.reconfigurationCost(control_signals)*...
                    (control_signals-prev_control_signals);
            else
                cost_gradient=zeros(model.nr_control_signals,1);
            end
        end
        
        function c_prev=previousControlSignal(model)
            if isempty(model.observations.control_signals)
                c_prev=zeros(model.nr_control_signals,1);
            else
                c_prev=model.observations.control_signals(end,:);
            end
        end
        
    end 
end