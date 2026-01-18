function [F_all, feas, SSNR] = opt_jsc_WOA( ...
    H_comm, sigmasq_ue, SINR_min, ...
    sensing_beamsteering, sens_streams, sigmasq_radar, ...
    Pmax, woa_iter, woa_agents, F_init)

% =====================================================
% WOA for Cell-free ISAC Beamforming (FINAL VERSION)
% =====================================================

%% Dimensions
F_size = size(F_init);
Nvar   = numel(F_init);
D      = 2 * Nvar;
U      = size(H_comm,2);

%% WOA params
nWhales = woa_agents;
MaxIter = woa_iter;

lb = -ones(D,1);
ub =  ones(D,1);

%% Encode initial solution
Fv = F_init(:);
x0 = [real(Fv); imag(Fv)];

X = zeros(nWhales, D);
for i = 1:nWhales
    X(i,:) = x0.' + 0.05*randn(1,D);
end

f_best = inf;

%% ================= WOA LOOP =================
for t = 1:MaxIter

    a = 2 - 2*t/MaxIter;

    for i = 1:nWhales

        %% Decode whale
        x = X(i,:).';
        F_vec = x(1:Nvar) + 1j*x(Nvar+1:end);
        F = reshape(F_vec, F_size);

        %% Power constraint
        pow = real(sum(abs(F(:)).^2));
        if pow > Pmax
            F = sqrt(Pmax/pow)*F;
        end

        %% Fitness
        [fval, feas_i, ssnr_i] = fitness_ISAC( ...
            F, H_comm, sigmasq_ue, SINR_min, ...
            sensing_beamsteering, sens_streams, sigmasq_radar, U);

        if fval < f_best
            f_best = fval;
            X_best = X(i,:);
            feas   = feas_i;
            SSNR   = ssnr_i;
        end
    end

    %% Update whales
    for i = 1:nWhales

        r1 = rand; r2 = rand;
        A = 2*a*r1 - a;
        C = 2*r2;

        p = rand;
        l = -1 + 2*rand;

        if p < 0.5
            if abs(A) < 1
                Dvec = abs(C*X_best - X(i,:));
                X(i,:) = X_best - A*Dvec;
            else
                j = randi(nWhales);
                Dvec = abs(C*X(j,:) - X(i,:));
                X(i,:) = X(j,:) - A*Dvec;
            end
        else
            Dvec = abs(X_best - X(i,:));
            X(i,:) = Dvec.*exp(l).*cos(2*pi*l) + X_best;
        end

        X(i,:) = max(min(X(i,:), ub'), lb');
    end
end

%% Output
x = X_best.';
F_vec = x(1:Nvar) + 1j*x(Nvar+1:end);
F_all = reshape(F_vec, F_size);

end

% =====================================================
% FITNESS FUNCTION (CELL-FREE SAFE)
% =====================================================
function [fval, feas, SSNR] = fitness_ISAC( ...
    F, H_comm, sigmasq_ue, SINR_min, ...
    sensing_beamsteering, sens_streams, sigmasq_radar, U)

feas = 1;
SINR = zeros(U,1);

num_ant = size(F,1);   % M_t * N_t

for u = 1:U

    % ===== CORRECT CHANNEL & BEAM =====
    hu = reshape(H_comm(u,:,:), num_ant, 1);
    fu = F(:,u);

    signal = abs(hu' * fu)^2;
    interf = 0;

    for k = 1:(U + sens_streams)
        if k ~= u
            interf = interf + abs(hu' * F(:,k))^2;
        end
    end

    SINR(u) = signal / (interf + sigmasq_ue);

    if SINR(u) < SINR_min
        feas = 0;
    end
end

% ===== Radar SSNR =====
R = F * F';
a = sensing_beamsteering(:);
SSNR = real(a' * R * a) / (sigmasq_radar * sens_streams);

% ===== Objective =====
penalty = sum(max(0, SINR_min - SINR).^2);
fval = -SSNR + 1e3 * penalty;

end