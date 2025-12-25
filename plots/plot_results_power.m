%% Collect the results
clear all;

[res_table, legend_list] = load_results('output', 'power');

%% Prepare Legends
for i = 3:size(res_table, 2)
    current_name = legend_list{i};

    % ---- Metaheuristic algorithms ----
    if strcmp(current_name, 'WOA')
        legend_list{i} = 'WOA (Meta-heuristic)';
        continue;
    elseif strcmp(current_name, 'CWOA')
        legend_list{i} = 'Chaotic WOA';
        continue;
    elseif strcmp(current_name, 'WOA-PSO')
        legend_list{i} = 'WOA–PSO Hybrid';
        continue;
    end

    % ---- Baselines & JSC ----
    txt = strsplit(current_name, '-');

    if length(txt) >= 1
        tf = strsplit(txt{1}, '+');

        if strcmp(tf{1}, 'JSC')
            legend_list{i} = 'JSC Beam Optimization';
        elseif length(tf) >= 2
            legend_list{i} = [tf{1}, ' Sensing - ', tf{2}, ' Comm'];
        else
            legend_list{i} = current_name;
        end
    else
        legend_list{i} = current_name;
    end
end


%% Plot the results
set_default_plot;

res_mat_mean = grpstats(res_table, 'Power-ratio');
power_ratio = res_mat_mean{:, 1};
res_mat_mean = res_mat_mean{:, 2:end}; % Remove power ratio

line_style = ["-", ":", "-.", "--", ":"];
marker = ['x', 'o', "square", 'd', '|'];
style_counter = 0;

figure;
for i = 3:size(res_mat_mean, 2)
    if rem(i, 2) == 1
        style_counter = max(1, rem(style_counter + 1, length(line_style)));
        subplot(2, 1, 1);
        title('Communication');
        ylabel('Min Comm SINR');
    else
        subplot(2, 1, 2);
        title('Sensing');
        ylabel('Target SNR')
        xlabel('Communication Power Ratio (\rho)');
    end
    plot(power_ratio, res_mat_mean(:, i), ...
        LineStyle=line_style(style_counter), ...
        Marker=marker(style_counter), ....
        DisplayName=legend_list{i})
    hold on;
    grid on;
end
legend;
