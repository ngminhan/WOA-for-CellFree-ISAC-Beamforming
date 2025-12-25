function [F_star, feasible, SSNR_opt] = opt_jsc_WOA( ...
    H_comm, sigmasq_comm, gamma_req, sensing_beamsteering, sens_streams, sigmasq_sens, ...
    P_all, max_iter, search_agents, F_init) 

    if nargin < 8 || isempty(max_iter), max_iter = 100; end
    if nargin < 9 || isempty(search_agents), search_agents = 30; end

    [U, M, N] = size(H_comm);
    H_st = reshape(H_comm, U, []);
    num_antennas = M * N;
    num_streams = U + sens_streams;
    dim = num_antennas * num_streams;

    % Sensing matrix
    a = reshape(sensing_beamsteering, 1, []);
    A_sens = a' * a;

    % Per-AP selection matrices D_m
    D_matrices = cell(M, 1);
    for m = 1:M
        if N > 1
            diag_idx = m : M : (M*N);
        else
            diag_idx = (m-1)*N + (1:N);
        end
        
        D_tmp = zeros(num_antennas, num_antennas);
        D_tmp(diag_idx, diag_idx) = eye(N);
        D_matrices{m} = D_tmp;
    end

    % ---- 1) Initialization ----
    % Random initialization (scaled per AP to satisfy power)
    X = (randn(search_agents, dim) + 1i * randn(search_agents, dim)) * sqrt(P_all / num_streams);
    for i = 1:search_agents
        X(i, :) = project_per_ap(X(i, :), num_antennas, num_streams, P_all, D_matrices);
    end

    if nargin >= 10 && ~isempty(F_init)
        X_init = reshape(F_init, 1, dim);
        X_init = project_per_ap(X_init, num_antennas, num_streams, P_all, D_matrices);
        X(1, :) = X_init;
    end

    % Evaluate initial best (maximize fitness)
    best_pos = X(1,:);
    best_fit = fitness(best_pos, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
    
    % Check random agent 
    for i = 2:search_agents
        fi = fitness(X(i,:), H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
        if fi > best_fit
            best_fit = fi;
            best_pos = X(i,:);
        end
    end

    % ---- 2) Vanilla WOA main loop ----
    b = 1;

    for t = 1:max_iter
        a_woa = 2 - 2 * (t-1) / (max_iter-1); 

        for i = 1:search_agents
            r1 = rand(); r2 = rand();
            A = 2 * a_woa * r1 - a_woa;  
            C = 2 * r2;               

            p = rand();
            l = -1 + 2*rand();          

            Xi = X(i,:);

            if p < 0.5
                if abs(A) < 1
                    % Encircling prey (exploitation)
                    D = abs(C * best_pos - Xi);
                    X_new = best_pos - A * D;
                else
                    % Search for prey (exploration) - random from whole population
                    rand_idx = randi(search_agents);
                    X_rand = X(rand_idx,:);
                    D = abs(C * X_rand - Xi);
                    X_new = X_rand - A * D;
                end
            else
                % Spiral updating
                Dp = abs(best_pos - Xi);
                X_new = Dp * exp(b*l) * cos(2*pi*l) + best_pos;
            end

            X_new = clip_complex(X_new, 5 * sqrt(P_all/num_streams));

            % Project to satisfy per-AP power (repair step)
            X_new = project_per_ap(X_new, num_antennas, num_streams, P_all, D_matrices);

            % Replace if better (greedy selection - common in practice)
            f_new = fitness(X_new, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
            f_old = fitness(Xi,    H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);

            if f_new > f_old
                X(i,:) = X_new;
                if f_new > best_fit
                    best_fit = f_new;
                    best_pos = X_new;
                end
            end
        end
    end

    % ---- 3) Final decode ----
    F_star = reshape(best_pos, num_antennas, num_streams);

    % compute final feasibility + SSNR (without penalties)
    [SSNR_opt, vio_sinr, vio_pow] = eval_raw(F_star, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
    feasible = (vio_sinr < 1e-3) && (vio_pow < 1e-3);
end

% ================= Helper functions =================

function x = clip_complex(x, max_abs)
    mag = abs(x);
    idx = mag > max_abs;
    x(idx) = x(idx) .* (max_abs ./ mag(idx));
end

function X_proj = project_per_ap(X_vec, num_antennas, num_streams, P_max, D_mats)
    F = reshape(X_vec, num_antennas, num_streams);
    F_sum = F * F';
    for m = 1:length(D_mats)
        p_m = real(trace(D_mats{m} * F_sum));
        if p_m > P_max
            scale = sqrt(P_max / p_m);
            diag_idx = find(diag(D_mats{m}));
            F(diag_idx, :) = F(diag_idx, :) * scale;
            F_sum = F * F';
        end
    end
    X_proj = reshape(F, 1, []);
end

function fit = fitness(X_vec, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams)
    F = reshape(X_vec, [], num_streams);
    [ssnr, vio_sinr, vio_pow] = eval_raw(F, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams);

    % Deb-style feasibility rule: ưu tiên nghiệm khả thi, sau đó tối đa hóa SSNR
    if vio_sinr > 1e-9 || vio_pow > 1e-9
        fit = -(vio_sinr + vio_pow);
    else
        fit = ssnr;
    end
end

function [ssnr, vio_sinr, vio_pow] = eval_raw(F, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams)
    % Objective: sensing term
    F_sum = F * F';
    obj_sensing = 0;
    for m = 1:length(D_mats)
        obj_sensing = obj_sensing + trace(D_mats{m} * A * D_mats{m} * F_sum);
    end
    ssnr = real(obj_sensing * sigma_sens);

    % SINR violations (sum of relative violations)
    vio_sinr = 0;
    for u = 1:U
        h_u = H(u,:);
        signal = abs(h_u * F(:,u))^2;

        inter = 0;
        for k = 1:num_streams
            if k ~= u
                inter = inter + abs(h_u * F(:,k))^2;
            end
        end

        sinr_u = signal / (inter + sigma_sq);
        if sinr_u < gamma * 0.999  
            vio_sinr = vio_sinr + (gamma - sinr_u)/gamma;
        end
    end

    % Power violation: sum over AP of exceed ratios
    vio_pow = 0;
    for m = 1:length(D_mats)
        p_m = real(trace(D_mats{m} * F_sum));
        if p_m > P_max
            vio_pow = vio_pow + (p_m - P_max)/P_max;
        end
    end
end
