classdef MaSDM_GA < Algorithm

    %------------------------------- Copyright --------------------------------
    % Copyright (c) Yanchi Li. You are free to use the MToP for research
    % purposes. All publications which use this platform should acknowledge
    % the use of "MToP" or "MTO-Platform" and cite as "Y. Li, W. Gong, F. Ming,
    % T. Zhang, S. Li, and Q. Gu, MToP: A MATLAB Optimization Platform for
    % Evolutionary Multitasking, 2023, arXiv:2312.08134"
    %--------------------------------------------------------------------------

    properties (SetAccess = private)
        MuC = 2
        MuM = 5
        KTN = 5
        Delta0 = 0.5
        Gap = 5
        Lambda0 = 0.5
        ParaMin = 0.05
        ParaMax = 0.95
        eta = 30
        split  = 50;
    end

    methods
        function Parameter = getParameter(Algo)
            Parameter = {'MuC: Simulated Binary Crossover', num2str(Algo.MuC), ...
                'MuM: Polynomial Mutation', num2str(Algo.MuM), ...
                'KTN: Knowledge Transfer Tasks Num', num2str(Algo.KTN), ...
                'Delta0: Initial delta', num2str(Algo.Delta0), ...
                'Gap: Parameter update gap', num2str(Algo.Gap), ...
                'Lambda0: Initial lambda', num2str(Algo.Lambda0), ...
                'ParaMin: Lower bound of parameter', num2str(Algo.ParaMin), ...
                'ParaMax: Upper bound of parameter', num2str(Algo.ParaMax), ...
                'eta: Parameter update generations', num2str(Algo.eta), ...
                'split: Number of intervals', num2str(Algo.split)};
        end

        function Algo = setParameter(Algo, Parameter)
            i = 1;
            Algo.MuC      = str2double(Parameter{i}); i = i + 1;
            Algo.MuM      = str2double(Parameter{i}); i = i + 1;
            Algo.KTN      = str2double(Parameter{i}); i = i + 1;
            Algo.Delta0   = str2double(Parameter{i}); i = i + 1;
            Algo.Gap    =  str2double(Parameter{i});  i = i + 1;
            Algo.Lambda0   = str2double(Parameter{i}); i = i + 1;
            Algo.ParaMin = str2double(Parameter{i}); i = i + 1;
            Algo.ParaMax = str2double(Parameter{i}); i = i + 1;
            Algo.eta     = str2double(Parameter{i}); i = i + 1;
            Algo.split   = str2double(Parameter{i}); 
        end

        function run(Algo, Prob)
            % Initialization
            population = Initialization(Algo, Prob, Individual);
            matrix_Q   = ones(Prob.T, Prob.T);
            matrix_R   = zeros(Prob.T, Prob.T);
            delta = Algo.Delta0 * ones(1, Prob.T);

            Ng = zeros(1, Prob.T);   
            Nb = zeros(1, Prob.T);   
            Nm = zeros(1, Prob.T);  

            lambda = Algo.Lambda0 * ones(1, Prob.T);

            % --------- Compute matrix_Q ---------
            featureMats = cell(1, Prob.T);     
            for t = 1:Prob.T
                featureMats{t} = Algo.ExtractTaskFeatureMatrix(Prob, t, Algo.split);
            end
            for i = 1:Prob.T
                for j = i+1:Prob.T
                    qij = Algo.ComputeTaskSimilarity(featureMats{i}, featureMats{j});
                    matrix_Q(i, j) = qij;
                    matrix_Q(j, i) = qij;
                end
                matrix_Q(i, i) = 0;
            end


            while Algo.notTerminated(Prob, population)

                % --------- Update matrix_R every 100 generations ---------
                if mod(Algo.Gen, Algo.eta) == 0
                    for t = 1:Prob.T
                        [~, best_index] = min([population{t}.Objs]);
                        a_x   = population{t}(best_index);
                        newa_x = a_x.Obj;

                        for k = 1:Prob.T
                            if k == t
                                continue;
                            end
                            [~, best_index_k] = min([population{k}.Objs]);
                            b_x    = population{k}(best_index_k);
                            newb_x = Algo.Evaluation(b_x, Prob, t);
                            bb     = newb_x.Obj;
                            x = (newa_x - bb) / (abs(newa_x) + 1e-12);
                            y = 0.5 * tanh(x + 0.5);
                            matrix_R(t, k) = y;

                            [~, worst_index] = max([population{t}.Objs]);
                            neww_x = population{t}(worst_index).Obj;
                            if bb < neww_x
                                population{t}(worst_index) = newb_x;
                            end
                        end
                    end
                end

                % --------- Generation ---------
                [offspring, transferFlag] = Algo.Generation(population, matrix_Q, matrix_R, delta, lambda);

                % --------- Evaluation + Environmental Selection ---------
                for t = 1:Prob.T
                    offspring{t} = Algo.Evaluation(offspring{t}, Prob, t);

                    oldN = length(population{t});
                    population{t} = [population{t}, offspring{t}];

                    [~, rank] = sort([population{t}.Objs]);
                    survivor_idx = rank(1:Prob.N);
                    offspring_survivor_idx = survivor_idx(survivor_idx > oldN) - oldN;

                    for s = 1:length(offspring_survivor_idx)
                        k = offspring_survivor_idx(s);

                        if transferFlag{t}(k) == 0
                            Ng(t) = Ng(t) + 1;
                        end

                        if transferFlag{t}(k) == 1
                            Nb(t) = Nb(t) + 1;
                        elseif transferFlag{t}(k) == 2
                            Nm(t) = Nm(t) + 1;
                        end
                    end

                    population{t} = population{t}(survivor_idx);
                end

                % --------- Update delta/lambda every Gap generations ---------
                if mod(Algo.Gen, Algo.Gap) == 0
                    for t = 1:Prob.T
                        if Nb(t) + Nm(t) > 0
                            delta(t) = Nb(t) / (Nb(t) + Nm(t));
                            delta(t) = min(max(delta(t), Algo.ParaMin), Algo.ParaMax);
                        end
                    end

                    for t = 1:Prob.T
                        aux_idx = [1:t-1, t+1:Prob.T];
                        valid_idx = aux_idx(matrix_Q(t, aux_idx) > 0);
                        score_aux = matrix_Q(t, valid_idx) .* exp(matrix_R(t, valid_idx));
                        p_t = score_aux / (sum(score_aux) + 1e-12);
                        S_t = -sum(p_t .* log(p_t + 1e-12)) / log(numel(valid_idx));

                        G_t = Ng(t) / (Ng(t) + Nb(t) + Nm(t));
                        lambda(t) = 0.5 * S_t + 0.5 * G_t;
                        lambda(t) = min(max(lambda(t), Algo.ParaMin), Algo.ParaMax);
                    end
                    Ng(:) = 0;
                    Nb(:) = 0;
                    Nm(:) = 0;
                end

            end
        end

        function [offspring, transferFlag] = Generation(Algo, population, matrix_Q, matrix_R, delta, lambda)
            offspring    = cell(1, length(population));
            transferFlag = cell(1, length(population));


            for t = 1:length(population)
                popSize = length(population{t});
                indorder = randperm(popSize);
                transferFlag{t} = zeros(1, popSize);

                [~, sorted_idx] = sort([population{t}.Objs]);
                elite_num = max(1, round(0.2 * length(population{t})));
                elite_idx = sorted_idx(1:elite_num);
                eliteDecs = vertcat(population{t}(elite_idx).Dec);
                mean_t = mean(eliteDecs, 1);
                avg_dist_t = mean(sqrt(sum((eliteDecs - mean_t).^2, 2)));


                count = 1;
                for i = 1:ceil(popSize / 2)
                    p1 = indorder(i);
                    p2 = indorder(i + fix(popSize / 2));

                    offspring{t}(count)     = population{t}(p1);
                    offspring{t}(count + 1) = population{t}(p2);

                    % ======================================================
                    % inter-task evolution
                    % ======================================================
                    if rand() < lambda(t)
                        transferFlag{t}(count)     = 0;
                        transferFlag{t}(count + 1) = 0;

                        [offspring{t}(count).Dec, offspring{t}(count + 1).Dec] = ...
                            GA_Crossover(population{t}(p1).Dec, population{t}(p2).Dec, Algo.MuC);

                        offspring{t}(count).Dec     = GA_Mutation(offspring{t}(count).Dec, Algo.MuM);
                        offspring{t}(count + 1).Dec = GA_Mutation(offspring{t}(count + 1).Dec, Algo.MuM);

                    else
                        knowledge_task_num = Algo.KTN;
                        task_scores = -inf(1, length(population));
                        
                        for j = 1:length(population)
                            if j == t
                                continue;
                            end
                            if matrix_Q(t, j) > 0
                                task_scores(j) = matrix_Q(t, j) * exp(matrix_R(t, j));
                            end
                        end
                        [~, index] = sort(task_scores, 'descend');
                        index = index(isfinite(task_scores(index)));
                        knowledge_task_num = min(knowledge_task_num, numel(index));
                        ass_tasks = index(1:knowledge_task_num);

                        if rand() < delta(t)
                            % ------------------------------------------------
                            % Strategy 1: Inter-task evolution with geometric constraint based on traditional operations
                            % ------------------------------------------------
                            transferFlag{t}(count)     = 1;
                            transferFlag{t}(count + 1) = 1;

                            
                            archive_corrected = population{t}(p1).Dec;

                            for j = 1:knowledge_task_num
                                rt = ass_tasks(j);
                                newpopulation = population{rt};

                                [~, i_best] = min([newpopulation.Objs]);
                                archive = newpopulation(i_best).Dec;
                                dist = norm(archive - mean_t);

                                if dist < avg_dist_t || dist < 1e-12
                                    archive_corrected(j, :) = archive;
                                else
                                    archive_corrected(j, :) = mean_t + (avg_dist_t / dist) * (archive - mean_t);
                                end

                            end

                            idx = randi(size(archive_corrected, 1));
                            target = archive_corrected(idx, :);
                            [c1, c2] = GA_Crossover(population{t}(p1).Dec, target, Algo.MuC);
                            offspring{t}(count).Dec = GA_Mutation(c1, Algo.MuM);
                            offspring{t}(count + 1).Dec = GA_Mutation(c2, Algo.MuM);

                        else
                            % ------------------------------------------------
                            % Strategy 2: Inter-task evolution based on isotropic Gaussian model
                            % ------------------------------------------------

                            transferFlag{t}(count)     = 2;
                            transferFlag{t}(count + 1) = 2;

                            for j = 1:knowledge_task_num
                                rt = ass_tasks(j);
                                newpopulation = population{rt};
                                [~, sorted_idx] = sort([newpopulation.Objs]);
                                elite_num = max(1, round(0.2 * length(newpopulation)));
                                elite_idx = sorted_idx(1:elite_num);
                                eliteDecs = vertcat(newpopulation(elite_idx).Dec);
                                mean_task{j} = mean(eliteDecs, 1);
                                avg_dist{j}  = mean(sqrt(sum((eliteDecs - mean_task{j}).^2, 2)));
                                tau2{j} = mean(sum((eliteDecs - mean_task{j}).^2, 2)) / size(eliteDecs, 2);
                                tau{j}  = sqrt(tau2{j});
                                Clu_scoresp1(j) = exp(-sum((population{t}(p1).Dec - mean_task{j}).^2 ./ (avg_dist{j}^2 + 1e-12)));
                                Clu_scoresp2(j) = exp(-sum((population{t}(p2).Dec - mean_task{j}).^2 ./ (avg_dist{j}^2 + 1e-12)));
                            end

                            for j = 1:knowledge_task_num
                                rt = ass_tasks(j);
                                fina_p1(j) = task_scores(rt) * Clu_scoresp1(j);
                                fina_p2(j) = task_scores(rt) * Clu_scoresp2(j);
                            end
                            [~,maxidx1] = max(fina_p1);
                            [~,maxidx2] = max(fina_p2);
                            offspring{t}(count).Dec = normrnd(mean_task{maxidx1}, tau{maxidx1});
                            offspring{t}(count + 1).Dec = normrnd(mean_task{maxidx2}, tau{maxidx2});
                        end
                    end

                    % Bound handling
                    for x = count:count + 1
                        offspring{t}(x).Dec(offspring{t}(x).Dec > 1) = 1;
                        offspring{t}(x).Dec(offspring{t}(x).Dec < 0) = 0;
                    end

                    count = count + 2;
                end
            end
        end


        function cosSim = ComputeTaskSimilarity(Algo, Mi, Mj)

            [n1, di] = size(Mi);
            [n2, dj] = size(Mj);

            if di == dj
                gi = Mi(:);
                gj = Mj(:);
                cosSim = (gi' * gj) / (norm(gi) * norm(gj) + 1e-12);
                return;
            end

            D = max(di, dj);          
            d_common = min(di, dj);  
            Mi_pad = zeros(n1, D);
            Mj_pad = zeros(n2, D);
            Mi_pad(:, 1:di) = Mi;
            Mj_pad(:, 1:dj) = Mj;
            Mi_pad = Mi_pad - mean(Mi_pad, 1);
            Mj_pad = Mj_pad - mean(Mj_pad, 1);
            [~, ~, Vi] = svd(Mi_pad, 'econ');
            [~, ~, Vj] = svd(Mj_pad, 'econ');

            d_common = min([d_common, size(Vi, 2), size(Vj, 2)]);
            Ai = Vi(:, 1:d_common);   
            Aj = Vj(:, 1:d_common);   
            Mij = Ai' * Aj;          
            Mi_proj = Mi_pad * Ai;           
            Mj_proj = Mj_pad * Aj * Mij';     
            gi = Mi_proj(:);
            gj = Mj_proj(:);

            cosSim = (gi' * gj) / (norm(gi) * norm(gj) + 1e-12);

        end

        function gamma = ExtractTaskFeatureMatrix(Algo, Prob, taskIdx, nInterval)

            d  = Prob.D(taskIdx);
            lb = Prob.Lb{taskIdx}(:)';
            ub = Prob.Ub{taskIdx}(:)';
            center = 0.5 * (lb + ub);
            M = zeros(nInterval, d);
            for dimIdx = 1:d
                grid = linspace(lb(dimIdx), ub(dimIdx), nInterval + 1);  
                X = repmat(center, nInterval + 1, 1);
                X(:, dimIdx) = grid(:);
                [obj, ~] = Prob.Fnc{taskIdx}(X);
                M(:, dimIdx) = diff(obj) ./ diff(grid(:));
            end
            gamma = reshape(M.', 1, []);
        end

    end
end