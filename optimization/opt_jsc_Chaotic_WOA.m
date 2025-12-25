function [F_star, feasible, SSNR_opt] = opt_jsc_Chaotic_WOA( ...
    H_comm, sigmasq_comm, gamma_req, sensing_beamsteering, sens_streams, sigmasq_sens, ...
    P_all, max_iter, search_agents, ~)

% Chaotic Whale Optimization Algorithm (Chaotic WOA)
% - Vanilla WOA mechanics
% - Logistic map for chaos
% - Constraint handling by penalty only
% - No warm start, no repair, no hybrid

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
        idx = (m-1)*N + (1:N);
        D = zeros(num_ant);
        D(idx,idx) = eye(N);
        D_matrices{m} = D;
    end

    % ---- 1) Random initialization (vanilla) ----
    X = (randn(search_agents,dim) + 1i*randn(search_agents,dim)) ...
        * sqrt(P_all/num_streams);

    % ---- Chaotic variable ----
    z = rand();      % chaotic seed
    mu = 4.0;        % logistic map parameter

    % Evaluate initial best
    best_pos = X(1,:);
    best_fit = fitness(best_pos);

    for i = 2:search_agents
        fi = fitness(X(i,:));
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

            % Greedy replacement
            if fitness(X_new) > fitness(Xi)
                X(i,:) = X_new;
                if fitness(X_new) > best_fit
                    best_fit = fitness(X_new);
                    best_pos = X_new;
                end
            end
        end
    end

    % ---- Final result ----
    F_star = reshape(best_pos, num_ant, num_streams);
    [SSNR_opt, vio_sinr, vio_pow] = eval_raw(F_star);
    feasible = (vio_sinr < 1e-4) && (vio_pow < 1e-4);

    % ===== Nested helper functions =====

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
