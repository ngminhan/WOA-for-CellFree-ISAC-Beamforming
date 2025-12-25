function [F_star, feasible, SSNR_opt] = opt_jsc_WOA_PSO( ...
    H_comm, sigmasq_comm, gamma_req, sensing_beamsteering, sens_streams, sigmasq_sens, ...
    P_all, max_iter, search_agents, ~)

% WOA–PSO Hybrid Algorithm
% - WOA for global exploration
% - PSO for local exploitation
% - Penalty-based constraint handling (no Deb, no repair)

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

    % Per-AP matrices
    D_matrices = cell(M,1);
    for m = 1:M
        idx = (m-1)*N + (1:N);
        D = zeros(num_ant);
        D(idx,idx) = eye(N);
        D_matrices{m} = D;
    end

    % ---- Initialization ----
    X = (randn(search_agents,dim) + 1i*randn(search_agents,dim)) ...
        * sqrt(P_all/num_streams);
    V = zeros(search_agents, dim);

    pbest = X;
    pbest_fit = zeros(search_agents,1);

    for i = 1:search_agents
        pbest_fit(i) = fitness(X(i,:));
    end

    [best_fit, idx] = max(pbest_fit);
    gbest = pbest(idx,:);

    % PSO parameters
    w  = 0.7;
    c1 = 1.5;
    c2 = 1.5;
    b  = 1;

    % ---- Main loop ----
    for t = 1:max_iter
        a_woa = 2 - 2*(t-1)/(max_iter-1);

        for i = 1:search_agents
            % ---------- WOA update ----------
            r1 = rand(); r2 = rand(); p = rand();
            l = -1 + 2*rand();

            A = 2*a_woa*r1 - a_woa;
            C = 2*r2;

            Xi = X(i,:);

            if p < 0.5
                if abs(A) < 1
                    D = abs(C*gbest - Xi);
                    X_woa = gbest - A*D;
                else
                    j = randi(search_agents);
                    Xr = X(j,:);
                    D = abs(C*Xr - Xi);
                    X_woa = Xr - A*D;
                end
            else
                Dp = abs(gbest - Xi);
                X_woa = Dp * exp(b*l) * cos(2*pi*l) + gbest;
            end

            % ---------- PSO refinement ----------
            V(i,:) = w*V(i,:) ...
                   + c1*rand()*(pbest(i,:) - Xi) ...
                   + c2*rand()*(gbest - Xi);

            X_new = X_woa + V(i,:);

            % Soft bound (avoid explosion)
            X_new = clip_complex(X_new, 5*sqrt(P_all/num_streams));

            f_new = fitness(X_new);

            % Update particle
            if f_new > pbest_fit(i)
                pbest(i,:) = X_new;
                pbest_fit(i) = f_new;
            end

            if f_new > best_fit
                best_fit = f_new;
                gbest = X_new;
            end

            X(i,:) = X_new;
        end
    end

    % ---- Final result ----
    F_star = reshape(gbest, num_ant, num_streams);
    [SSNR_opt, vio_sinr, vio_pow] = eval_raw(F_star);
    feasible = (vio_sinr < 1e-4) && (vio_pow < 1e-4);

    % ===== Helper functions =====
    function fit = fitness(X_vec)
        F = reshape(X_vec, [], num_streams);
        [ssnr, vio_sinr, vio_pow] = eval_raw(F);
        fit = ssnr - 1e6*vio_sinr - 1e6*vio_pow;
    end

    function [ssnr, vio_sinr, vio_pow] = eval_raw(F)
        Fsum = F*F';
        ssnr = 0;
        for mm = 1:M
            ssnr = ssnr + trace(D_matrices{mm}*A_sens*D_matrices{mm}*Fsum);
        end
        ssnr = real(ssnr * sigmasq_sens);

        vio_sinr = 0;
        for u = 1:U
            h = H_st(u,:);
            s = abs(h*F(:,u))^2;
            i0 = 0;
            for k = 1:num_streams
                if k~=u, i0 = i0 + abs(h*F(:,k))^2; end
            end
            sinr = s/(i0+sigmasq_comm);
            if sinr < gamma_req
                vio_sinr = vio_sinr + (gamma_req-sinr)/gamma_req;
            end
        end

        vio_pow = 0;
        for mm = 1:M
            pm = real(trace(D_matrices{mm}*Fsum));
            if pm > P_all
                vio_pow = vio_pow + (pm-P_all)/P_all;
            end
        end
    end
end

function x = clip_complex(x, maxabs)
    mag = abs(x);
    idx = mag > maxabs;
    x(idx) = x(idx) .* (maxabs ./ mag(idx));
end
