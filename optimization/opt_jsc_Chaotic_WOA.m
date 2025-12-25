function [F_star, feasible, SSNR_opt] = opt_jsc_Chaotic_WOA( ...
    H_comm, sigmasq_comm, gamma_req, sensing_beamsteering, sens_streams, sigmasq_sens, ...
    P_all, max_iter, search_agents, F_init) % <--- SỬA: Nhận tham số F_init

% Chaotic Whale Optimization Algorithm (Chaotic WOA)
% - Vanilla WOA mechanics + Logistic map for chaos
% - Warm start capability added

    if nargin < 8 || isempty(max_iter), max_iter = 100; end
    if nargin < 9 || isempty(search_agents), search_agents = 30; end

    [U, M, N] = size(H_comm);
    H_st = reshape(H_comm, U, []);
    num_ant = M * N;
    num_streams = U + sens_streams;
    dim = num_ant * num_streams;

    % Sensing matrix
    a = reshape(sensing_beamsteering, 1, []);
    A_sens = a' * a;

    % Per-AP selection matrices
    D_matrices = cell(M,1);
    for m = 1:M
        if N > 1
            diag_idx = m : M : (M*N); % Logic Interleaved
        else
            diag_idx = (m-1)*N + (1:N);
        end
        D = zeros(num_ant);
        D(diag_idx, diag_idx) = eye(N);
        D_matrices{m} = D;
    end
    
    % ---- 1) Random initialization (vanilla) ----
    X = (randn(search_agents,dim) + 1i*randn(search_agents,dim)) ...
        * sqrt(P_all/num_streams);
    for i = 1:search_agents
        X(i, :) = project_per_ap(X(i, :), num_ant, num_streams, P_all, D_matrices);
    end

    % --- FIX: Apply Warm Start ---
    if nargin >= 10 && ~isempty(F_init)
        % Gán nghiệm khởi tạo tốt cho agent đầu tiên, chiếu về miền công suất
        X_init = reshape(F_init, 1, dim);
        X_init = project_per_ap(X_init, num_ant, num_streams, P_all, D_matrices);
        X(1, :) = X_init;
    end

    % ---- Chaotic variable ----
    z = rand();      % chaotic seed
    mu = 4.0;        % logistic map parameter

    % Evaluate initial best
    best_pos = X(1,:);
    best_fit = fitness(best_pos, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);

    for i = 2:search_agents
        fi = fitness(X(i,:), H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
        if fi > best_fit
            best_fit = fi;
            best_pos = X(i,:);
        end
    end

    % ---- 2) Chaotic WOA main loop ----
    b = 1;

    for t = 1:max_iter
        a_woa = 2 - 2*(t-1)/(max_iter-1);

        for i = 1:search_agents
            % ---- Chaotic random numbers ----
            z = mu*z*(1-z); r1 = z;
            z = mu*z*(1-z); r2 = z;
            z = mu*z*(1-z); p  = z;
            z = mu*z*(1-z); l  = -1 + 2*z;

            A = 2*a_woa*r1 - a_woa;
            C = 2*r2;

            Xi = X(i,:);

            if p < 0.5
                if abs(A) < 1
                    D = abs(C*best_pos - Xi);
                    X_new = best_pos - A*D;
                else
                    z = mu*z*(1-z);
                    j = floor(z*search_agents) + 1;
                    % Bảo đảm index hợp lệ
                    if j < 1, j=1; end; if j > search_agents, j=search_agents; end
                    
                    Xr = X(j,:);
                    D = abs(C*Xr - Xi);
                    X_new = Xr - A*D;
                end
            else
                Dp = abs(best_pos - Xi);
                X_new = Dp * exp(b*l) * cos(2*pi*l) + best_pos;
            end

            % Soft bound (not repair)
            X_new = clip_complex(X_new, 5*sqrt(P_all/num_streams));

            % Project to satisfy per-AP power (repair step)
            X_new = project_per_ap(X_new, num_ant, num_streams, P_all, D_matrices);

            % Greedy replacement
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

    % ---- Final result ----
    F_star = reshape(best_pos, num_ant, num_streams);
    [SSNR_opt, vio_sinr, vio_pow] = eval_raw(F_star, H_st, sigmasq_comm, gamma_req, A_sens, sigmasq_sens, P_all, D_matrices, U, num_streams);
    feasible = (vio_sinr < 1e-3) && (vio_pow < 1e-3);
end

% ===== Nested helper functions =====
function x = clip_complex(x, maxabs)
    mag = abs(x);
    idx = mag > maxabs;
    x(idx) = x(idx) .* (maxabs ./ mag(idx));
end

function X_proj = project_per_ap(X_vec, num_ant, num_streams, P_max, D_mats)
    F = reshape(X_vec, num_ant, num_streams);
    Fsum = F * F';
    for m = 1:length(D_mats)
        p_m = real(trace(D_mats{m} * Fsum));
        if p_m > P_max
            scale = sqrt(P_max / p_m);
            diag_idx = find(diag(D_mats{m}));
            F(diag_idx, :) = F(diag_idx, :) * scale;
            Fsum = F * F';
        end
    end
    X_proj = reshape(F, 1, []);
end

function fit = fitness(X_vec, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams)
    F = reshape(X_vec, [], num_streams);
    [ssnr, vio_sinr, vio_pow] = eval_raw(F, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams);

    if vio_sinr > 1e-9 || vio_pow > 1e-9
        fit = -(vio_sinr + vio_pow);
    else
        fit = ssnr;
    end
end

function [ssnr, vio_sinr, vio_pow] = eval_raw(F, H, sigma_sq, gamma, A, sigma_sens, P_max, D_mats, U, num_streams)
    Fsum = F*F';
    ssnr = 0;
    for mm = 1:length(D_mats)
        ssnr = ssnr + trace(D_mats{mm}*A*D_mats{mm}*Fsum);
    end
    ssnr = real(ssnr * sigma_sens);

    vio_sinr = 0;
    for u = 1:U
        h = H(u,:);
        s = abs(h*F(:,u))^2;
        i0 = 0;
        for k = 1:num_streams
            if k~=u, i0 = i0 + abs(h*F(:,k))^2; end
        end
        sinr = s/(i0+sigma_sq);
        if sinr < gamma * 0.999
            vio_sinr = vio_sinr + (gamma-sinr)/gamma;
        end
    end

    vio_pow = 0;
    for mm = 1:length(D_mats)
        pm = real(trace(D_mats{mm}*Fsum));
        if pm > P_max
            vio_pow = vio_pow + (pm-P_max)/P_max;
        end
    end
end
