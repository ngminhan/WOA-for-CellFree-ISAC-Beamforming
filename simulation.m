%% Simulation
function results = simulation(params, output_filename)

    save_filename = output_filename;
    results = {};

    for rep = 1:params.repetitions
        fprintf('\n Repetition %i:', rep);

        %% Generate positions
        [UE_pos, AP_pos, target_pos] = generate_positions( ...
            params.T, params.U, params.M_t, ...
            params.geo.line_length, params.geo.target_y, ...
            params.geo.UE_y, params.geo.min_dist, params.geo.max_dist);

        results{rep}.P_comm_ratio = params.P_comm_ratio;
        results{rep}.AP     = AP_pos;
        results{rep}.UE     = UE_pos;
        results{rep}.Target = target_pos;

        %% Channel generation
        H_comm = LOS_channel(AP_pos, UE_pos, params.N_t);

        if params.N_t == 1 && ismatrix(H_comm)
             H_comm = reshape(H_comm, params.U, params.M_t, params.N_t);
        end

        %% Sensing beamsteering
        [sensing_angle, ~] = compute_angle_dist(AP_pos, target_pos);
        sensing_beamsteering = beamsteering(sensing_angle.', params.N_t);

        F_sensing_CB_norm = sensing_beamsteering * sqrt(1/params.N_t);
        F_sensing_NS_norm = beam_nulling(H_comm, sensing_beamsteering);

        %% Loop over power ratios
        for p_i = 1:length(params.P_comm_ratio)

            P_comm    = params.P * params.P_comm_ratio(p_i);
            P_sensing = params.P * (1 - params.P_comm_ratio(p_i));

            F_sensing_CB = F_sensing_CB_norm * sqrt(P_sensing);
            F_sensing_NS = F_sensing_NS_norm * sqrt(P_sensing);

            solution_counter = 1;

            %% ================= BASELINES =================

            % NS + RZF
            F_star_RZF = beam_regularized_zeroforcing( ...
                H_comm, P_comm, params.sigmasq_ue) * sqrt(P_comm);

            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_star_RZF, params.sigmasq_ue, ...
                sensing_beamsteering, F_sensing_NS, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'NS+RZF';
            solution_counter = solution_counter + 1;

            % NS + OPT
            wrapped_objective = @(gamma) opt_comm_SOCP_vec( ...
                H_comm, params.sigmasq_ue, P_comm, F_sensing_NS, gamma);
            [F_star_SOCP_NS, SINR_min_SOCP_NS] = bisection_SINR( ...
                params.bisect.low, params.bisect.high, params.bisect.tol, wrapped_objective);

            SOCP_NS_feasible = ~(isempty(SINR_min_SOCP_NS) || sum(abs(F_star_SOCP_NS(:))) < 1e-8);

            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_star_SOCP_NS, params.sigmasq_ue, ...
                sensing_beamsteering, F_sensing_NS, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'NS+OPT';
            results{rep}.power{p_i}{solution_counter}.min_SINR_opt = SINR_min_SOCP_NS;
            results{rep}.power{p_i}{solution_counter}.feasible = SOCP_NS_feasible;
            solution_counter = solution_counter + 1;

            % CB + OPT
            wrapped_objective = @(gamma) opt_comm_SOCP_vec( ...
                H_comm, params.sigmasq_ue, P_comm, F_sensing_CB, gamma);
            [F_star_SOCP_CB, SINR_min_SOCP_CB] = bisection_SINR( ...
                params.bisect.low, params.bisect.high, params.bisect.tol, wrapped_objective);

            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_star_SOCP_CB, params.sigmasq_ue, ...
                sensing_beamsteering, F_sensing_CB, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'CB+OPT';
            results{rep}.power{p_i}{solution_counter}.min_SINR_opt = SINR_min_SOCP_CB;
            solution_counter = solution_counter + 1;

            %% ================= JSC (SDP) =================
            sens_streams = 1;

            [Q_jsc, feasible_jsc, SSNR_jsc] = opt_jsc_SDP( ...
                H_comm, params.sigmasq_ue, SINR_min_SOCP_NS, ...
                sensing_beamsteering, sens_streams, ...
                params.sigmasq_radar_rcs, params.P);

            [F_jsc_comm, F_jsc_sensing] = SDP_beam_extraction(Q_jsc, H_comm);

            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_jsc_comm, params.sigmasq_ue, ...
                sensing_beamsteering, F_jsc_sensing, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'JSC-SDP';
            results{rep}.power{p_i}{solution_counter}.feasible = feasible_jsc;
            results{rep}.power{p_i}{solution_counter}.SSNR_opt = SSNR_jsc;
            solution_counter = solution_counter + 1;

            %% ================= METAHEURISTICS =================
            woa_iter   = 100;
            woa_agents = 20;

            % ---------------------------------------------------------
            % [1] WARM START
            % ---------------------------------------------------------
            F_init_woa = zeros(params.M_t*params.N_t, params.U + sens_streams);

            if exist('SOCP_NS_feasible', 'var') && SOCP_NS_feasible
                F_ref_comm = F_star_SOCP_NS;
                target_sinr_ref = SINR_min_SOCP_NS;
            else
                F_ref_comm = F_star_RZF;
                target_sinr_ref = 0; 
            end

            [d1, d2, d3] = size(F_ref_comm);
            num_ant_total = params.M_t * params.N_t;
            
            if d1 == num_ant_total && d2 == params.U
                for u = 1:params.U, F_init_woa(:,u) = F_ref_comm(:, u); end
            elseif d1 == params.U && d2 == num_ant_total
                for u = 1:params.U, F_init_woa(:,u) = F_ref_comm(u, :).'; end
            elseif d1 == params.U && d3 == params.N_t
                for u = 1:params.U, F_init_woa(:,u) = reshape(F_ref_comm(u,:,:), [], 1); end
            else
                 % Fallback
                 for u = 1:params.U, F_init_woa(:,u) = reshape(F_ref_comm(u,:), [], 1); end
            end

            % Sensing part
            if size(F_sensing_NS, 1) == sens_streams && size(F_sensing_NS, 2) == params.M_t
                 for s = 1:sens_streams, F_init_woa(:, params.U+s) = reshape(F_sensing_NS(s,:,:), [], 1); end
            elseif size(F_sensing_NS, 2) == sens_streams
                 for s = 1:sens_streams, F_init_woa(:, params.U+s) = F_sensing_NS(:, s); end
            else
                 for s = 1:sens_streams, F_init_woa(:, params.U+s) = reshape(F_sensing_NS(s,:,:), [], 1); end
            end

            % ---------------------------------------------------------
            % [2] DEBUG LOG
            % ---------------------------------------------------------
            fprintf('\n--- DEBUG WARM START (Rep %d) ---\n', rep);
            fprintf('F_init size: %s | Target SINR: %.4f | SOCP feasible: %d\n', mat2str(size(F_init_woa)), target_sinr_ref, exist('SOCP_NS_feasible','var') && SOCP_NS_feasible);
            
            [U_db, M_db, N_db] = size(H_comm);
            if N_db > 1, H_db = reshape(H_comm, U_db, []); else, H_db = H_comm; end
            
            vio_sinr_db = 0;
            for u = 1:params.U
                h_u = H_db(u,:);
                f_u = F_init_woa(:,u);
                
                % Signal
                sig = abs(h_u * f_u)^2;
                
                % Interference
                inter = 0;
                for k = 1:(params.U+sens_streams)
                    if k~=u
                        inter = inter + abs(h_u * F_init_woa(:,k))^2;
                    end
                end
                
                sinr_val = sig / (inter + params.sigmasq_ue);
                
                % So sánh với target
                if target_sinr_ref > 0
                    if sinr_val < target_sinr_ref * 0.99
                        diff = (target_sinr_ref - sinr_val)/target_sinr_ref;
                        vio_sinr_db = vio_sinr_db + diff;
                        fprintf('  User %d: SINR=%.2f (Target %.2f) [FAIL]\n', u, sinr_val, target_sinr_ref);
                    else
                        fprintf('  User %d: SINR=%.2f [OK]\n', u, sinr_val);
                    end
                end
            end
            
            if vio_sinr_db < 1e-3
                fprintf('=> WARM START SUCCESS! (Ready for optimization)\n');
            else
                fprintf('=> WARM START ISSUE: Total SINR violation = %.4f\n', vio_sinr_db);
            end
            fprintf('--------------------------------------\n');

            % ---------------------------------------------------------
            % [3] CALL METAHEURISTICS
            % ---------------------------------------------------------
            
            % ---- Vanilla WOA ----
            [F_all, feas, SSNR] = opt_jsc_WOA( ...
                H_comm, params.sigmasq_ue, SINR_min_SOCP_NS, ...
                sensing_beamsteering, sens_streams, params.sigmasq_radar_rcs, ...
                params.P, woa_iter, woa_agents, F_init_woa); 

            [F_comm, F_sens] = split_F(F_all, params);
            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_comm, params.sigmasq_ue, ...
                sensing_beamsteering, F_sens, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'WOA';
            results{rep}.power{p_i}{solution_counter}.feasible = feas;
            results{rep}.power{p_i}{solution_counter}.SSNR_opt = SSNR;
            solution_counter = solution_counter + 1;

            % ---- Chaotic WOA ----
            [F_all, feas, SSNR] = opt_jsc_Chaotic_WOA( ...
                H_comm, params.sigmasq_ue, SINR_min_SOCP_NS, ...
                sensing_beamsteering, sens_streams, params.sigmasq_radar_rcs, ...
                params.P, woa_iter, woa_agents, F_init_woa);

            [F_comm, F_sens] = split_F(F_all, params);
            results{rep}.power{p_i}{solution_counter} = ...
                compute_metrics(H_comm, F_comm, params.sigmasq_ue, ...
                sensing_beamsteering, F_sens, params.sigmasq_radar_rcs);
            results{rep}.power{p_i}{solution_counter}.name = 'CWOA';
            results{rep}.power{p_i}{solution_counter}.feasible = feas;
            results{rep}.power{p_i}{solution_counter}.SSNR_opt = SSNR;
            solution_counter = solution_counter + 1;

            % ---- WOA–PSO ----
            [F_all, feas, SSNR] = opt_jsc_WOA_PSO( ...
                H_comm, params.sigmasq_ue, SINR_min_SOCP_NS, ...
                sensing_beamsteering, sens_streams, params.sigmasq_radar_rcs, ...
                params.P, woa_iter, woa_agents, F_init_woa);

                [F_comm, F_sens] = split_F(F_all, params);
                results{rep}.power{p_i}{solution_counter} = ...
                    compute_metrics(H_comm, F_comm, params.sigmasq_ue, ...
                    sensing_beamsteering, F_sens, params.sigmasq_radar_rcs);
                
                results{rep}.power{p_i}{solution_counter}.name = 'WOA-PSO';
                results{rep}.power{p_i}{solution_counter}.feasible = feas;
                results{rep}.power{p_i}{solution_counter}.SSNR_opt = SSNR;
                solution_counter = solution_counter + 1;
        end
    end

    %% Save results
    output_folder = './output/';
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    save(fullfile(output_folder, [save_filename,'.mat']));
end

function [F_comm, F_sens] = split_F(F_all, params)
    F_comm = zeros(params.U, params.M_t, params.N_t);
    for u = 1:params.U
        F_comm(u,:,:) = reshape(F_all(:,u), params.M_t, params.N_t);
    end

    S = size(F_all,2) - params.U;
    F_sens = zeros(S, params.M_t, params.N_t);
    for s = 1:S
        F_sens(s,:,:) = reshape(F_all(:,params.U+s), params.M_t, params.N_t);
    end
end